from __future__ import annotations

from ..config import get_settings
from ..llm import structured_completion
from ..logging_utils import narrate
from ..prompts import INTAKE_SYSTEM
from ..schemas import GraphState, WorkloadSpec


def intake(state: GraphState) -> dict:
    """Normalize the request into a WorkloadSpec.

    The primary path is a workload YAML already parsed by the CLI; the
    free-text path turns a plain-English migration request into a spec.
    """
    if state.get("workload"):
        w = state["workload"]
        log = narrate(
            f"intake: loaded workload '{w.name}' — {len(w.samples)} samples, "
            f"baseline {w.baseline.name}, "
            f"targets: quality>={w.constraints.min_quality_score}"
            + (f", p95<={w.constraints.max_p95_latency_s}s" if w.constraints.max_p95_latency_s else "")
            + (f", cost -{w.constraints.min_cost_reduction_pct}%" if w.constraints.min_cost_reduction_pct else "")
        )
        return {"decision_log": [log], "iterations": 0}

    s = get_settings()
    raw = state.get("raw_request", "")
    if not raw:
        raise ValueError("No workload YAML and no free-text request provided.")
    w = structured_completion(s.supervisor_model, WorkloadSpec, INTAKE_SYSTEM, raw)
    log = narrate(f"intake: parsed free-text request into workload '{w.name}' ({len(w.samples)} samples)")
    return {"workload": w, "decision_log": [log], "iterations": 0}
