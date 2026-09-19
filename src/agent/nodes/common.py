from __future__ import annotations

from typing import Optional

from ..schemas import CandidateRun, CandidateSpec, GraphState, JudgeVerdict, RunConfig, WorkloadSpec


def latest_run(state: GraphState, model_id: str) -> Optional[CandidateRun]:
    runs = [r for r in state.get("runs", []) if r.model_id == model_id]
    return max(runs, key=lambda r: r.attempt) if runs else None


def latest_verdict(state: GraphState, model_id: str) -> Optional[JudgeVerdict]:
    vs = [v for v in state.get("verdicts", []) if v.model_id == model_id]
    return max(vs, key=lambda v: v.attempt) if vs else None


def spec_for(state: GraphState, model_id: str) -> CandidateSpec:
    for c in state.get("candidates", []):
        if c.model_id == model_id:
            return c
    return CandidateSpec(model_id=model_id)


def eval_payload(
    candidate: CandidateSpec,
    workload: WorkloadSpec,
    attempt: int,
    config: Optional[RunConfig] = None,
) -> dict:
    return {
        "candidate": candidate,
        "workload": workload,
        "attempt": attempt,
        "run_config": config or RunConfig(),
    }
