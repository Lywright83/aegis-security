#!/usr/bin/env python3
"""
agent.py -- Personal AI Agent (v1.1)

Your self-hosted, open-source AI assistant.
Run: python agent.py

Commands:
  (type a task)           -- Agent plans and executes it
  /tools                  -- List available tools
  /skills                 -- List installed community skills
  /install <path>         -- Install a skill from file or GitHub URL
  /remove <name>          -- Remove an installed skill
  /personality            -- Show current personality settings
  /preferences            -- Show adaptive preference summary
  /feedback <type>        -- Give feedback: too_long, too_short, too_formal, too_casual, good
  /config                 -- Show active configuration summary
  /heartbeat start        -- Start heartbeat (background daemon)
  /heartbeat start-fg     -- Start heartbeat (foreground, live output)
  /heartbeat stop         -- Stop the heartbeat
  /heartbeat status       -- Show heartbeat status
  /heartbeat log          -- Show recent heartbeat history
  /heartbeat once         -- Run one heartbeat cycle right now
  /help                   -- Show this help
  /quit                   -- Exit the agent
"""

import json
import re
import sys
import time
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich import box

from config_loader import load_config, get_llm_config
from llm_client import call_llm
from tool_registry import discover_tools, execute_tool, list_tools, get_tools_for_llm
from personality import build_system_prompt, PreferenceTracker
from timeline import MissionTimeline
from logger import setup_logger
from marketplace import (
    install_from_local, install_from_github,
    list_installed_skills, remove_skill
)
from heartbeat import HeartbeatScheduler, run_heartbeat_cycle

console = Console()

# -- Tool-call parser ------------------------------------------------

TOOL_CALL_PATTERN = re.compile(
    r"```tool\s*\n(\{.*?\})\s*\n```",
    re.DOTALL,
)


def parse_tool_calls(response: str) -> list[dict]:
    """Extract tool call JSON blocks from the LLM response."""
    calls = []
    for match in TOOL_CALL_PATTERN.finditer(response):
        try:
            data = json.loads(match.group(1))
            if "tool" in data:
                calls.append(data)
        except json.JSONDecodeError:
            continue
    return calls


def strip_tool_blocks(response: str) -> str:
    """Remove tool call blocks from response to get the plain text."""
    return TOOL_CALL_PATTERN.sub("", response).strip()


# -- Core agent loop -------------------------------------------------

def run_task(task: str, config: dict, llm_config: dict,
              tracker: PreferenceTracker, logger) -> str:
    """
    Execute a single task:
    1. Build system prompt with personality + tools
    2. Send task to LLM
    3. Parse and execute any tool calls
    4. Return results to LLM for summary
    5. Render the timeline
    """
    timeline = MissionTimeline(task)
    max_steps = config.get("safety", {}).get("max_steps_per_task", 10)

    # Detect if this is a heartbeat task (for quieter output)
    is_heartbeat = task.startswith("[HEARTBEAT:")

    # Build system prompt
    system_prompt = build_system_prompt(config, tracker)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": task},
    ]

    logger.info(f"{'[HB] ' if is_heartbeat else ''}Task received: {task[:100]}")

    # Planning step
    plan_step = timeline.add_step("Analysing task and planning approach", "llm_call")
    plan_step.start()

    try:
        response = call_llm(messages, llm_config)
    except Exception as e:
        plan_step.finish("failed", str(e))
        logger.error(f"LLM call failed: {e}")
        if not is_heartbeat:
            timeline.render_terminal()
        return f"LLM call failed: {e}"

    plan_step.finish("success")
    logger.debug(f"LLM response: {response[:200]}...")

    messages.append({"role": "assistant", "content": response})

    # Tool execution loop
    step_count = 0
    while step_count < max_steps:
        tool_calls = parse_tool_calls(response)
        if not tool_calls:
            break

        for tc in tool_calls:
            step_count += 1
            if step_count > max_steps:
                logger.warning(f"Max steps ({max_steps}) reached -- stopping.")
                break

            tool_name = tc.get("tool", "unknown")
            tool_args = tc.get("args", {})

            # Timeline step
            step = timeline.add_step(
                f"Execute tool: {tool_name}",
                tool_name,
            )
            step.start()
            logger.info(f"Calling tool: {tool_name} with args: {tool_args}")

            # Execute
            result = execute_tool(tool_name, tool_args, config)
            failed = result.startswith("!") or "failed" in result.lower()[:20]
            step.finish("failed" if failed else "success", result[:200])
            logger.info(f"   Tool result: {result[:150]}...")

            # Feed result back to LLM
            messages.append({
                "role": "user",
                "content": f"Tool '{tool_name}' returned:\n{result}\n\nContinue with the task or provide the final answer.",
            })

            # Get next LLM response
            follow_step = timeline.add_step("Processing tool result", "llm_call")
            follow_step.start()
            try:
                response = call_llm(messages, llm_config)
            except Exception as e:
                follow_step.finish("failed", str(e))
                break
            follow_step.finish("success")

            messages.append({"role": "assistant", "content": response})

    # Final answer
    final_text = strip_tool_blocks(response)

    # Record for adaptive personality
    tracker.record_interaction(task, len(final_text))

    # Render timeline (skip for daemon-mode heartbeat tasks)
    if not is_heartbeat:
        timeline.render_terminal()

    # Save mission data
    missions_dir = config.get("logging", {}).get("missions_dir", "missions")
    json_path = timeline.save_json(missions_dir)
    html_path = timeline.export_html(missions_dir)
    logger.info(f"Mission saved: {json_path}")
    logger.info(f"Timeline HTML: {html_path}")

    if not is_heartbeat:
        console.print(f"\n  Mission log: [dim]{json_path}[/dim]")
        console.print(f"  Timeline:   [dim]{html_path}[/dim]\n")

    return final_text


# -- CLI commands ----------------------------------------------------

# Global heartbeat scheduler reference (set in main)
_heartbeat_scheduler: HeartbeatScheduler | None = None


def handle_command(command: str, config: dict, tracker: PreferenceTracker,
                    llm_config: dict, logger) -> bool:
    """
    Handle slash commands. Returns True if the agent should continue,
    False if it should exit.
    """
    global _heartbeat_scheduler

    parts = command.strip().split()
    cmd = parts[0].lower()
    arg = " ".join(parts[1:]) if len(parts) > 1 else ""

    if cmd == "/quit":
        if _heartbeat_scheduler:
            _heartbeat_scheduler.stop()
        console.print("[yellow]Goodbye![/yellow]")
        return False

    elif cmd == "/help":
        console.print(__doc__)

    elif cmd == "/tools":
        tools = list_tools()
        if not tools:
            console.print("[yellow]No tools registered.[/yellow]")
        else:
            from rich.table import Table
            table = Table(title="Available Tools", box=box.ROUNDED,
                          border_style="cyan")
            table.add_column("Name", style="yellow")
            table.add_column("Description")
            table.add_column("Risk", justify="center")
            table.add_column("Source", style="dim")
            for t in tools:
                table.add_row(t["name"], t["description"],
                              t["risk_level"], t.get("source", ""))
            console.print(table)

    elif cmd == "/skills":
        list_installed_skills()

    elif cmd == "/install":
        if not arg:
            console.print("[red]Usage: /install <path_or_github_url>[/red]")
        elif arg.startswith(("http://", "https://")):
            result = install_from_github(arg)
            console.print(result)
            discover_tools(config)
        else:
            result = install_from_local(arg)
            console.print(result)
            discover_tools(config)

    elif cmd == "/remove":
        if not arg:
            console.print("[red]Usage: /remove <skill_name>[/red]")
        else:
            result = remove_skill(arg)
            console.print(result)

    elif cmd == "/personality":
        p_conf = config.get("personality", {})
        console.print(Panel(
            f"Active preset: [cyan]{p_conf.get('active_preset', 'professional')}[/cyan]\n"
            f"Adaptive learning: [cyan]{p_conf.get('adaptive', {}).get('enabled', False)}[/cyan]",
            title="Personality",
            border_style="cyan",
        ))

    elif cmd == "/preferences":
        console.print(f"[cyan]{tracker.get_summary()}[/cyan]")

    elif cmd == "/feedback":
        valid = ["good", "too_long", "too_short", "too_formal", "too_casual"]
        if arg not in valid:
            console.print(f"[red]Usage: /feedback <{'|'.join(valid)}>[/red]")
        else:
            tracker.record_interaction("(manual feedback)", 0, arg)
            console.print(f"[green]Feedback recorded: {arg}[/green]")

    elif cmd == "/config":
        llm = config.get("llm", {})
        hb = config.get("heartbeat", {})
        console.print(Panel(
            f"Agent: [cyan]{config.get('agent', {}).get('name', 'Agent')}[/cyan]\n"
            f"LLM provider: [cyan]{llm.get('active_provider', '?')}[/cyan]\n"
            f"Model: [cyan]{llm.get(llm.get('active_provider', ''), {}).get('model', '?')}[/cyan]\n"
            f"Read-only mode: [cyan]{config.get('safety', {}).get('read_only_mode', True)}[/cyan]\n"
            f"Heartbeat: [magenta]{hb.get('mode', 'daemon')} / {hb.get('interval_minutes', 30)}min[/magenta]\n"
            f"Tools enabled: [cyan]{', '.join(t['name'] for t in list_tools())}[/cyan]",
            title="Configuration",
            border_style="cyan",
        ))

    elif cmd == "/heartbeat":
        _handle_heartbeat_command(arg, config, llm_config, tracker, logger)

    else:
        console.print(f"[red]Unknown command: {cmd}. Type /help for options.[/red]")

    return True


def _handle_heartbeat_command(sub_cmd: str, config: dict, llm_config: dict,
                                tracker: PreferenceTracker, logger):
    """Handle /heartbeat subcommands."""
    global _heartbeat_scheduler

    # Lazy-init the scheduler
    if _heartbeat_scheduler is None:
        hb_config = config.get("heartbeat", {})
        interval = hb_config.get("interval_minutes", 30)
        _heartbeat_scheduler = HeartbeatScheduler(
            config=config,
            llm_config=llm_config,
            tracker=tracker,
            logger=logger,
            run_task_fn=run_task,
            interval_minutes=interval,
            foreground=False,
        )

    if sub_cmd == "start":
        result = _heartbeat_scheduler.start_background()
        console.print(f"[magenta]{result}[/magenta]")

    elif sub_cmd == "start-fg":
        console.print("[magenta]Starting heartbeat in foreground (Ctrl+C to stop)...[/magenta]\n")
        _heartbeat_scheduler.foreground = True
        _heartbeat_scheduler.start_foreground()

    elif sub_cmd == "stop":
        result = _heartbeat_scheduler.stop()
        console.print(f"[magenta]{result}[/magenta]")

    elif sub_cmd == "status":
        status = _heartbeat_scheduler.status()
        console.print(Panel(status, title="Heartbeat Status", border_style="magenta"))

    elif sub_cmd == "log":
        _heartbeat_scheduler.show_log()

    elif sub_cmd == "once":
        console.print("[magenta]Running one heartbeat cycle...[/magenta]\n")
        run_heartbeat_cycle(
            config, llm_config, tracker, logger,
            run_task, foreground=True,
        )

    else:
        console.print(
            "[red]Usage: /heartbeat <start|start-fg|stop|status|log|once>[/red]\n"
            "  start    -- Start heartbeat in background (daemon)\n"
            "  start-fg -- Start heartbeat in foreground (live output)\n"
            "  stop     -- Stop the heartbeat\n"
            "  status   -- Show heartbeat status\n"
            "  log      -- Show recent heartbeat history\n"
            "  once     -- Run one heartbeat cycle right now"
        )


# -- Banner ----------------------------------------------------------

BANNER = """
[bold cyan]
    +=============================================+
    |          ARCHON  v1.1                        |
    |     Your Personal AI Agent Framework         |
    |                                              |
    |  Type a task, or /help for commands          |
    |  /heartbeat start  to enable background mode |
    +=============================================+
[/bold cyan]"""


# -- First boot walkthrough ------------------------------------------

def _run_cli_walkthrough(config: dict, agent_name: str):
    """Show a guided walkthrough on first launch. Skips on subsequent runs."""
    import time as _time
    flag_file = Path("personality/.first_boot_done")

    if flag_file.exists():
        console.print(Panel(
            f"[bold]Welcome back.[/bold] What are we working on?",
            title=f"{agent_name}",
            border_style="cyan",
        ))
        return

    # First boot -- full walkthrough
    console.print()
    console.print("[dim]Initializing systems...[/dim]")
    _time.sleep(0.8)
    console.print("[dim]Security shield: [green]ACTIVE[/green][/dim]")
    _time.sleep(0.5)
    console.print(f"[dim]Tools loaded: [cyan]{len(list_tools())}[/cyan] | Expertise layers: [cyan]3[/cyan][/dim]")
    _time.sleep(0.8)

    console.print()
    console.print(Panel(
        f"[bold]I'm online. Welcome to the Command Center.[/bold]\n\n"
        f"I'm {agent_name} -- your elite security operator, Fortune 500 strategist,\n"
        f"master trader, polyglot translator, and coder. All in one.\n\n"
        f"Let me show you what I can do.",
        title=f"{agent_name}",
        border_style="cyan",
        padding=(1, 2),
    ))

    _time.sleep(1)
    console.print(Panel(
        "[bold cyan]WHAT I CAN DO:[/bold cyan]\n\n"
        "[cyan]SECURITY[/cyan]    Analyze logs, MITRE ATT&CK mapping, IR playbooks,\n"
        "            threat intel, code review, exploit analysis\n\n"
        "[cyan]CODING[/cyan]      Write, run, and review code in any language.\n"
        "            Security-first. I find what attackers would find.\n\n"
        "[cyan]EMAIL[/cyan]       Gmail, Outlook, Yahoo, ProtonMail -- read, send,\n"
        "            reply, forward. All with your approval.\n\n"
        "[cyan]BUSINESS[/cyan]    Fortune 500 strategy, financial analysis,\n"
        "            market sizing, executive communication.\n\n"
        "[cyan]TRADING[/cyan]     Day trading, swing, commodities, crypto.\n"
        "            Technical + fundamental analysis. Educational only.\n\n"
        "[cyan]TRANSLATE[/cyan]   100+ languages with cultural context.",
        border_style="dim",
        padding=(1, 2),
    ))

    _time.sleep(1)
    console.print(Panel(
        "[bold cyan]COMMANDS:[/bold cyan]\n\n"
        "  [yellow]/tools[/yellow]            List all available tools\n"
        "  [yellow]/heartbeat start[/yellow]  Start background task scheduler\n"
        "  [yellow]/heartbeat once[/yellow]   Run one heartbeat cycle now\n"
        "  [yellow]/personality[/yellow]      Show personality settings\n"
        "  [yellow]/preferences[/yellow]      See what I've learned about you\n"
        "  [yellow]/feedback <type>[/yellow]  Tell me: too_long, too_short, good\n"
        "  [yellow]/config[/yellow]           Show configuration\n"
        "  [yellow]/help[/yellow]             Full command list\n"
        "  [yellow]/quit[/yellow]             Exit\n\n"
        "[bold cyan]SECURITY:[/bold cyan]\n\n"
        "  3-tier permissions: auto reads / OK for writes / password for deletes\n"
        "  Kill word: type it anywhere and I lock down instantly\n"
        "  Prompt injection shield: always active",
        border_style="dim",
        padding=(1, 2),
    ))

    _time.sleep(0.5)
    console.print(Panel(
        "[bold cyan]TRY THESE:[/bold cyan]\n\n"
        '  "Look up MITRE technique T1110"\n'
        '  "Analyze CVE-2021-44228"\n'
        '  "Create IR playbook templates"\n'
        '  "Translate hello to Japanese, Arabic, and Swahili"\n'
        '  "What makes a strong P&L statement?"\n'
        '  "Review this Python code for security issues"\n\n'
        "Or just talk to me. I'm here.\n\n"
        "[bold]Let's build something legendary.[/bold]",
        border_style="cyan",
        padding=(1, 2),
    ))

    # Mark walkthrough complete
    flag_file.parent.mkdir(parents=True, exist_ok=True)
    flag_file.write_text("1")
    console.print()


# -- Main ------------------------------------------------------------

def main():
    global _heartbeat_scheduler

    # Load config
    try:
        config = load_config("config.yaml")
    except FileNotFoundError as e:
        console.print(f"[red]{e}[/red]")
        sys.exit(1)

    # Setup
    logger = setup_logger(config)
    llm_config = get_llm_config(config)
    tracker = PreferenceTracker(
        config.get("personality", {})
              .get("adaptive", {})
              .get("preferences_file", "personality/preferences.json")
    )

    # Discover tools
    discover_tools(config)

    # Welcome
    console.print(BANNER)
    agent_name = config.get("agent", {}).get("name", "Agent")
    hb_config = config.get("heartbeat", {})
    console.print(f"  Provider: [cyan]{llm_config['provider']}[/cyan] | "
                   f"Model: [cyan]{llm_config.get('model', '?')}[/cyan] | "
                   f"Tools: [cyan]{len(list_tools())}[/cyan] | "
                   f"Heartbeat: [magenta]{hb_config.get('interval_minutes', 30)}min[/magenta]\n")

    # First boot walkthrough
    _run_cli_walkthrough(config, agent_name)

    # Auto-start heartbeat if configured as daemon
    if hb_config.get("enabled", False) and hb_config.get("mode") == "daemon":
        _heartbeat_scheduler = HeartbeatScheduler(
            config=config,
            llm_config=llm_config,
            tracker=tracker,
            logger=logger,
            run_task_fn=run_task,
            interval_minutes=hb_config.get("interval_minutes", 30),
            foreground=False,
        )
        result = _heartbeat_scheduler.start_background()
        console.print(f"  [magenta]{result}[/magenta]\n")

    # Interactive loop
    while True:
        try:
            user_input = console.input("[bold green]You >[/bold green] ").strip()
        except (KeyboardInterrupt, EOFError):
            if _heartbeat_scheduler:
                _heartbeat_scheduler.stop()
            console.print("\n[yellow]Goodbye![/yellow]")
            break

        if not user_input:
            continue

        # Handle slash commands
        if user_input.startswith("/"):
            should_continue = handle_command(
                user_input, config, tracker, llm_config, logger
            )
            if not should_continue:
                break
            continue

        # Run task
        console.print(f"\n[dim]Working on your task...[/dim]\n")
        try:
            result = run_task(user_input, config, llm_config, tracker, logger)
            console.print(Panel(
                result,
                title=f"{agent_name}",
                border_style="green",
                padding=(1, 2),
            ))
        except Exception as e:
            logger.error(f"Task failed: {e}", exc_info=True)
            console.print(f"[red]Task failed: {e}[/red]")


if __name__ == "__main__":
    main()
