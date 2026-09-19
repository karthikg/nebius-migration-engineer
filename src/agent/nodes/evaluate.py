from __future__ import annotations

import math
import time

from ..llm import chat_model
from ..logging_utils import narrate
from ..schemas import CandidateRun, CandidateSpec, RunConfig, SampleResult, WorkloadSpec
from ..tools import validate_output


def _p95(values: list[float]) -> float:
    if not values:
        return 0.0
    vs = sorted(values)
    k = max(0, math.ceil(0.95 * len(vs)) - 1)
    return vs[k]


def evaluate(payload: dict) -> dict:
    """Run every workload sample against one candidate model on Token Factory.

    Invoked via LangGraph `Send` — one instance per candidate, in parallel.
    The payload carries the candidate, its run config (which encodes any fix
    the supervisor ordered) and the attempt number.
    """
    candidate: CandidateSpec = payload["candidate"]
    workload: WorkloadSpec = payload["workload"]
    config: RunConfig = payload["run_config"]
    attempt: int = payload["attempt"]

    llm = chat_model(
        candidate.model_id,
        temperature=config.temperature,
        max_tokens=workload.max_output_tokens,
        json_mode=config.response_format_json,
        tags=["evaluate", candidate.model_id],
    )

    results: list[SampleResult] = []
    for i, sample in enumerate(workload.samples):
        prompt = workload.prompt_template.format(input=sample.input)
        if config.prompt_suffix:
            prompt += "\n\n" + config.prompt_suffix
        t0 = time.perf_counter()
        try:
            out = llm.invoke(prompt).content
            out = out if isinstance(out, str) else str(out)
            error = None
        except Exception as e:  # noqa: BLE001 - a failing sample is data, not a crash
            out, error = "", f"{type(e).__name__}: {e}"
        latency = time.perf_counter() - t0

        if workload.constraints.require_valid_json and not error:
            valid, missing, parse_err = validate_output(out, workload.constraints.required_keys)
            error = error or parse_err
        else:
            valid, missing = (not error), []
        results.append(
            SampleResult(
                sample_index=i, output=out, latency_s=round(latency, 3),
                json_valid=valid, missing_keys=missing, error=error,
            )
        )

    run = CandidateRun(
        model_id=candidate.model_id,
        attempt=attempt,
        config=config,
        results=results,
        p95_latency_s=round(_p95([r.latency_s for r in results]), 3),
        json_valid_count=sum(1 for r in results if r.json_valid),
        n_samples=len(results),
    )
    log = narrate(
        f"evaluate: {candidate.model_id} (attempt {attempt}, {config.describe()}): "
        f"JSON {run.json_valid_count}/{run.n_samples}, p95 {run.p95_latency_s:.2f}s"
    )
    return {"runs": [run], "decision_log": [log]}
