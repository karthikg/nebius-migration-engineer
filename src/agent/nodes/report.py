from __future__ import annotations

import json
import time
from typing import Optional

from ..config import get_settings
from ..llm import structured_completion
from ..logging_utils import narrate
from ..prompts import REPORT_SYSTEM
from ..schemas import (
    CandidateSummaryRow,
    GraphState,
    MigrationRecommendation,
    ReportNarrative,
)
from .common import latest_run, latest_verdict


def _build_rows(state: GraphState) -> list[CandidateSummaryRow]:
    cost_by_model = {c.model_id: c for c in state.get("costs", [])}
    rows = []
    for c in state.get("candidates", []):
        m = c.model_id
        run = latest_run(state, m)
        if not run:
            continue
        verdict = latest_verdict(state, m)
        cost = cost_by_model.get(m)
        rows.append(CandidateSummaryRow(
            model_id=m,
            attempt=run.attempt,
            avg_quality=verdict.avg_score if verdict else None,
            json_valid=f"{run.json_valid_count}/{run.n_samples}",
            p95_latency_s=run.p95_latency_s,
            monthly_cost_usd=cost.monthly_cost_usd if cost else None,
            savings_pct=cost.savings_pct if cost else None,
            status=state.get("statuses", {}).get(m, "unknown"),
        ))
    return rows


def _pick_winner(state: GraphState, rows: list[CandidateSummaryRow]) -> Optional[str]:
    """Deterministic winner selection among accepted candidates:
    highest savings first, quality as tiebreak. The LLM writes the narrative;
    the numbers pick the winner."""
    accepted = [r for r in rows if r.status == "accepted"]
    if not accepted:
        return None
    accepted.sort(key=lambda r: (-(r.savings_pct or -1), -(r.avg_quality or 0)))
    return accepted[0].model_id


def report(state: GraphState) -> dict:
    s = get_settings()
    w = state["workload"]
    rows = _build_rows(state)
    winner = _pick_winner(state, rows)
    verdict = "MIGRATE" if winner else "DONT_MIGRATE"

    evidence = {
        "verdict": verdict,
        "recommended_model": winner,
        "endpoint_base_url": s.base_url,
        "workload": w.model_dump(exclude={"samples", "prompt_template"}),
        "evidence_table": [r.model_dump() for r in rows],
        "decision_log": state.get("decision_log", []),
    }
    narrative: ReportNarrative = structured_completion(
        s.supervisor_model, ReportNarrative, REPORT_SYSTEM,
        json.dumps(evidence, indent=2, default=str),
    )
    rec = MigrationRecommendation(
        verdict=verdict,
        recommended_model=winner,
        confidence=narrative.confidence,
        summary=narrative.summary,
        migration_notes=narrative.migration_notes,
        caveats=narrative.caveats,
        rows=rows,
    )

    # Persist evidence: report.md + full state dump under runs/<timestamp>/.
    ts = time.strftime("%Y%m%d-%H%M%S")
    out_dir = s.runs_dir / f"{ts}-{w.name}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.md").write_text(render_markdown(state, rec))
    (out_dir / "state.json").write_text(json.dumps(
        {k: v for k, v in state.items()},
        indent=2,
        default=lambda o: o.model_dump() if hasattr(o, "model_dump") else str(o),
    ))
    log = narrate(f"report: {verdict}" + (f" -> {winner}" if winner else "") + f" (evidence in {out_dir})")
    return {"recommendation": rec, "report_path": str(out_dir / "report.md"), "decision_log": [log]}


def render_markdown(state: GraphState, rec: MigrationRecommendation) -> str:
    w = state["workload"]
    lines = [
        f"# Migration report — {w.name}",
        "",
        f"**Verdict: {rec.verdict}**"
        + (f" → `{rec.recommended_model}`" if rec.recommended_model else "")
        + f"  (confidence: {rec.confidence})",
        "",
        rec.summary,
        "",
        "## Candidates evaluated",
        "",
        "| Model | Attempt | Quality /5 | JSON valid | p95 latency | Monthly cost | Savings | Status |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rec.rows:
        cost = f"${r.monthly_cost_usd:,.0f}" if r.monthly_cost_usd is not None else "unknown"
        savings = f"{r.savings_pct:+.1f}%" if r.savings_pct is not None else "—"
        quality = f"{r.avg_quality:.2f}" if r.avg_quality is not None else "—"
        lines.append(
            f"| {r.model_id} | {r.attempt} | {quality} | {r.json_valid} | "
            f"{r.p95_latency_s:.2f}s | {cost} | {savings} | {r.status} |"
        )
    if state.get("costs"):
        baseline_cost = state["costs"][0].baseline_monthly_cost_usd
        lines += ["", f"Baseline: {w.baseline.name} at ${baseline_cost:,.0f}/month"]
    lines += [
        "",
        "## Migration notes",
        "",
        rec.migration_notes,
        "",
        "## Caveats",
        "",
    ]
    lines += [f"- {c}" for c in rec.caveats]
    lines += [
        "",
        "## Reasoning trail (agent decision log)",
        "",
    ]
    lines += [f"- {entry}" for entry in state.get("decision_log", [])]
    lines += ["", "---", "*Generated by tf-migration-agent against live Nebius Token Factory endpoints.*", ""]
    return "\n".join(lines)
