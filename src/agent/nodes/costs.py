from __future__ import annotations

from ..logging_utils import narrate
from ..schemas import CostReport, GraphState
from ..tools import pricing_table


def _monthly_cost(reqs_per_day: int, in_tok: int, out_tok: int,
                  in_price: float, out_price: float) -> float:
    monthly_requests = reqs_per_day * 30
    return round(monthly_requests * (in_tok * in_price + out_tok * out_price) / 1_000_000, 2)


def costs(state: GraphState) -> dict:
    """Deterministic cost analysis: pricing table + volume profile -> $/month."""
    w = state["workload"]
    pricing = pricing_table()
    v = w.volume
    baseline_cost = _monthly_cost(
        v.requests_per_day, v.avg_input_tokens, v.avg_output_tokens,
        w.baseline.input_price_per_mtok, w.baseline.output_price_per_mtok,
    )

    reports: list[CostReport] = []
    log: list[str] = []
    seen: set[str] = set()
    for run in state.get("runs", []):
        m = run.model_id
        if m in seen:
            continue
        seen.add(m)
        p = pricing.get(m)
        if not p:
            reports.append(CostReport(
                model_id=m, monthly_cost_usd=None,
                baseline_monthly_cost_usd=baseline_cost,
                savings_pct=None, meets_budget=None,
            ))
            log.append(narrate(f"costs: {m}: pricing unknown (not in live catalog or fallback table) — flagged"))
            continue
        cost = _monthly_cost(
            v.requests_per_day, v.avg_input_tokens, v.avg_output_tokens,
            p["input_price_per_mtok"], p["output_price_per_mtok"],
        )
        savings = round(100 * (1 - cost / baseline_cost), 1) if baseline_cost else None
        target = w.constraints.min_cost_reduction_pct
        meets = (savings >= target) if (savings is not None and target is not None) else None
        reports.append(CostReport(
            model_id=m, monthly_cost_usd=cost,
            baseline_monthly_cost_usd=baseline_cost,
            savings_pct=savings, meets_budget=meets,
        ))
        log.append(narrate(
            f"costs: {m}: ${cost:,.0f}/mo vs baseline ${baseline_cost:,.0f}/mo "
            f"({savings:+.1f}%, {p.get('source', 'unknown source')}) "
            f"=> {'meets' if meets else 'MISSES' if meets is not None else 'no'} budget target"
        ))
    return {"costs": reports, "decision_log": log}
