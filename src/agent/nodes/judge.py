from __future__ import annotations

from statistics import mean

from ..config import get_settings
from ..llm import structured_completion
from ..logging_utils import narrate
from ..prompts import JUDGE_SYSTEM
from ..schemas import CandidateRun, GraphState, JudgeScores, JudgeVerdict, WorkloadSpec

_TRUNC = 900


def _clip(text: str) -> str:
    return text if len(text) <= _TRUNC else text[:_TRUNC] + " …[truncated]"


def _judge_user(w: WorkloadSpec, run: CandidateRun) -> str:
    parts = [f"Task: {w.task_description}\n"]
    for r in run.results:
        sample = w.samples[r.sample_index]
        parts.append(
            f"--- SAMPLE {r.sample_index} ---\n"
            f"INPUT:\n{_clip(sample.input)}\n\n"
            f"REFERENCE OUTPUT:\n{_clip(sample.reference_output)}\n\n"
            f"CANDIDATE OUTPUT:\n{_clip(r.output) if r.output else '(empty / request error: ' + str(r.error) + ')'}\n"
        )
    parts.append(
        f"\nScore all {len(run.results)} samples (sample_index 0..{len(run.results)-1})."
    )
    return "\n".join(parts)


def judge(state: GraphState) -> dict:
    """LLM-as-judge sub-agent: scores each new candidate run against the
    reference outputs. Runs once per round (fan-in after the evaluate fan-out)."""
    s = get_settings()
    w = state["workload"]
    already = {(v.model_id, v.attempt) for v in state.get("verdicts", [])}
    new_runs = [r for r in state.get("runs", []) if (r.model_id, r.attempt) not in already]

    verdicts: list[JudgeVerdict] = []
    log: list[str] = []
    for run in new_runs:
        parsed: JudgeScores = structured_completion(
            s.judge_model, JudgeScores, JUDGE_SYSTEM, _judge_user(w, run)
        )
        # Deterministic floor: an errored/empty sample is a 1 regardless of the judge.
        by_idx = {sc.sample_index: sc for sc in parsed.scores}
        for r in run.results:
            if (not r.output) and r.sample_index in by_idx:
                by_idx[r.sample_index].score = 1.0
        scores = sorted(by_idx.values(), key=lambda sc: sc.sample_index)
        avg = round(mean(sc.score for sc in scores), 2) if scores else 1.0
        verdicts.append(
            JudgeVerdict(
                model_id=run.model_id, attempt=run.attempt,
                avg_score=avg, scores=scores, summary=parsed.summary,
            )
        )
        log.append(narrate(
            f"judge: {run.model_id} (attempt {run.attempt}): quality {avg:.2f}/5 — {parsed.summary[:110]}"
        ))
    return {"verdicts": verdicts, "decision_log": log}
