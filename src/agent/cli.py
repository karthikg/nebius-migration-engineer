from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


def _load_workload(path: str):
    from .schemas import WorkloadSpec

    with open(path) as f:
        data = yaml.safe_load(f)
    return WorkloadSpec.model_validate(data)


def cmd_run(args: argparse.Namespace) -> int:
    from .graph import build_graph
    from .schemas import GraphState

    state: GraphState = {}
    if args.workload:
        state["workload"] = _load_workload(args.workload)
    elif args.prompt:
        state["raw_request"] = args.prompt
    else:
        console.print("[red]Provide a workload YAML or --prompt.[/red]")
        return 2
    if args.candidates:
        state["candidates_override"] = [m.strip() for m in args.candidates.split(",")]

    console.print(Panel.fit(
        "[bold]Migration Engineer[/bold] — LangGraph on Nebius Token Factory",
        border_style="cyan",
    ))
    app = build_graph()
    final = app.invoke(state, config={"recursion_limit": 100})

    rec = final.get("recommendation")
    if not rec:
        console.print("[red]No recommendation produced — see decision log above.[/red]")
        return 1

    console.print()
    verdict_style = "green" if rec.verdict == "MIGRATE" else "yellow"
    title = f"[bold {verdict_style}]{rec.verdict}[/bold {verdict_style}]"
    if rec.recommended_model:
        title += f" → [bold]{rec.recommended_model}[/bold]"
    console.print(Panel(rec.summary, title=title, subtitle=f"confidence: {rec.confidence}"))

    table = Table(title="Candidates evaluated")
    for col in ("Model", "Attempt", "Quality /5", "JSON", "p95", "Monthly cost", "Savings", "Status"):
        table.add_column(col)
    for r in rec.rows:
        table.add_row(
            r.model_id,
            str(r.attempt),
            f"{r.avg_quality:.2f}" if r.avg_quality is not None else "—",
            r.json_valid,
            f"{r.p95_latency_s:.2f}s",
            f"${r.monthly_cost_usd:,.0f}" if r.monthly_cost_usd is not None else "unknown",
            f"{r.savings_pct:+.1f}%" if r.savings_pct is not None else "—",
            r.status,
        )
    console.print(table)
    console.print(f"\n[bold]Migration notes:[/bold] {rec.migration_notes}")
    if rec.caveats:
        console.print("[bold]Caveats:[/bold]")
        for c in rec.caveats:
            console.print(f"  • {c}")
    console.print(f"\nFull report + evidence: [cyan]{final.get('report_path')}[/cyan]")
    return 0


def cmd_preflight(args: argparse.Namespace) -> int:
    from .config import get_settings
    from .preflight import markdown_matrix, render_table, run_preflight

    s = get_settings()
    models = args.models or [s.supervisor_model, s.judge_model, *s.default_candidate_pool]
    # de-dup, keep order
    models = list(dict.fromkeys(models))
    results = run_preflight(models)
    console.print()
    console.print(render_table(results))
    if args.markdown:
        console.print("\nMarkdown matrix (paste into README):\n")
        print(markdown_matrix(results))
    critical_fail = any(
        not results[m]["basic"][0] for m in models
    )
    return 1 if critical_fail else 0


def cmd_models(args: argparse.Namespace) -> int:
    from .tools import fetch_catalog

    ids = sorted(m["id"] for m in fetch_catalog())
    for i in ids:
        if not args.keyword or args.keyword.lower() in i.lower():
            console.print(i)
    console.print(f"\n[dim]{len(ids)} models in the live catalog[/dim]")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="tf-migrate",
        description="Migration Engineer — find and validate the best open model "
        "on Nebius Token Factory for your workload.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="Run a migration evaluation")
    p_run.add_argument("workload", nargs="?", help="Path to a workload YAML (see workloads/)")
    p_run.add_argument("--prompt", help="Free-text migration request instead of a YAML")
    p_run.add_argument("--candidates", help="Comma-separated model ids to pin (skips the planner's own pick)")
    p_run.set_defaults(fn=cmd_run)

    p_pre = sub.add_parser("preflight", help="Verify model capabilities on the live endpoint")
    p_pre.add_argument("-m", "--models", nargs="*", help="Model ids (default: supervisor, judge, candidate pool)")
    p_pre.add_argument("--markdown", action="store_true", help="Also print a markdown matrix")
    p_pre.set_defaults(fn=cmd_preflight)

    p_models = sub.add_parser("models", help="List the live Token Factory catalog")
    p_models.add_argument("keyword", nargs="?", help="Filter substring")
    p_models.set_defaults(fn=cmd_models)

    args = parser.parse_args()
    sys.exit(args.fn(args))


if __name__ == "__main__":
    main()
