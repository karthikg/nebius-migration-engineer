from __future__ import annotations

from typing import Optional

from ..config import get_settings
from ..llm import structured_completion
from ..logging_utils import narrate
from ..prompts import (
    JSON_HARDENING_SUFFIX,
    QUALITY_HARDENING_SUFFIX,
    SUPERVISOR_SYSTEM,
)
from ..schemas import GraphState, RunConfig, SupervisorRuling
from ..tools import model_features, probe_json_mode
from .common import eval_payload, latest_run, latest_verdict, spec_for


def _fix_to_config(fix: Optional[str]) -> RunConfig:
    if fix == "enforce_json_response_format":
        return RunConfig(response_format_json=True, prompt_suffix=JSON_HARDENING_SUFFIX)
    if fix == "strengthen_prompt":
        return RunConfig(prompt_suffix=QUALITY_HARDENING_SUFFIX)
    return RunConfig()


def _applied_fixes(state: GraphState, model_id: str) -> set[str]:
    fixes: set[str] = set()
    for r in state.get("runs", []):
        if r.model_id != model_id:
            continue
        if r.config.response_format_json:
            fixes.add("enforce_json_response_format")
        elif r.config.prompt_suffix:
            fixes.add("strengthen_prompt")
    return fixes


def _evidence(state: GraphState, active: list[str]) -> str:
    w = state["workload"]
    cost_by_model = {c.model_id: c for c in state.get("costs", [])}
    lines = [
        f"Workload: {w.name}",
        f"Constraints: {w.constraints.model_dump()}",
        f"Samples per run: {len(w.samples)}",
        "",
        "ACTIVE CANDIDATES (latest attempt):",
    ]
    for m in active:
        run = latest_run(state, m)
        verdict = latest_verdict(state, m)
        cost = cost_by_model.get(m)
        if not run:
            continue
        lines.append(
            f"- {m} | attempt {run.attempt} ({run.config.describe()}) | "
            f"JSON valid {run.json_valid_count}/{run.n_samples} | "
            f"p95 latency {run.p95_latency_s:.2f}s | "
            f"quality {verdict.avg_score if verdict else '?'}/5 "
            f"(judge: {verdict.summary if verdict else 'n/a'}) | "
            f"cost savings {f'{cost.savings_pct}%' if cost and cost.savings_pct is not None else 'unknown'} | "
            f"fixes already applied: {sorted(_applied_fixes(state, m)) or 'none'} | "
            f"endpoint-supported features: {model_features(m) or 'unknown'}"
        )
        if verdict:
            worst = min(verdict.scores, key=lambda s: s.score, default=None)
            if worst:
                lines.append(f"    worst sample: #{worst.sample_index} score {worst.score} — {worst.notes[:140]}")
    return "\n".join(lines)


def supervise(state: GraphState) -> dict:
    """The supervisor reviews all evidence and routes each candidate:
    accept, eliminate, or retry-with-fix. This is the reasoning core of the
    agent — every ruling lands in the decision log."""
    s = get_settings()
    w = state["workload"]
    iterations = state.get("iterations", 0) + 1
    statuses = dict(state.get("statuses", {}))
    active = [m for m, st in statuses.items() if st == "active"]
    log: list[str] = []
    pending: list[dict] = []

    if not active:
        return {"iterations": iterations, "pending": [], "decision_log": [narrate("supervise: no active candidates left")]}

    ruling: SupervisorRuling = structured_completion(
        s.supervisor_model,
        SupervisorRuling,
        SUPERVISOR_SYSTEM.format(
            iteration=iterations,
            max_iterations=s.max_iterations,
            n_samples=len(w.samples),
        ),
        _evidence(state, active),
    )
    log.append(narrate(f"supervise (round {iterations}): {ruling.commentary}"))

    decided: set[str] = set()
    for d in ruling.decisions:
        if d.model_id not in active or d.model_id in decided:
            continue
        decided.add(d.model_id)
        run = latest_run(state, d.model_id)
        action, fix = d.action, d.fix

        # Deterministic guardrails on top of the LLM's ruling.
        if action == "accept" and w.constraints.require_valid_json and run and run.json_valid_count < run.n_samples:
            action = "retry" if iterations < s.max_iterations else "eliminate"
            fix = fix or "enforce_json_response_format"
            log.append(narrate(
                f"supervise: OVERRIDE — accept for {d.model_id} rejected "
                f"(JSON {run.json_valid_count}/{run.n_samples}) -> {action}"
            ))
        if action == "retry" and iterations >= s.max_iterations:
            log.append(narrate(f"supervise: retry budget exhausted for {d.model_id} -> eliminate"))
            action = "eliminate"
        downgraded_from_json = False
        if action == "retry" and fix == "enforce_json_response_format" and "json_mode" not in model_features(d.model_id):
            # Catalog metadata says no json_mode — but metadata and behavior
            # can disagree, so ask the live endpoint before giving up on it.
            if probe_json_mode(d.model_id):
                log.append(narrate(
                    f"supervise: catalog metadata for {d.model_id} lacks json_mode, but a live probe "
                    f"honored response_format — trusting the probe"
                ))
            else:
                log.append(narrate(
                    f"supervise: OVERRIDE — {d.model_id} failed a live json_mode probe, "
                    f"downgrading to prompt-level JSON hardening"
                ))
                fix = "strengthen_prompt"
                downgraded_from_json = True
        if action == "retry" and fix in _applied_fixes(state, d.model_id):
            log.append(narrate(f"supervise: fix '{fix}' already tried on {d.model_id} -> eliminate"))
            action = "eliminate"

        if action == "accept":
            statuses[d.model_id] = "accepted"
            log.append(narrate(f"supervise: ACCEPT {d.model_id} — {d.reasoning}"))
        elif action == "eliminate":
            statuses[d.model_id] = "eliminated"
            log.append(narrate(f"supervise: ELIMINATE {d.model_id} — {d.reasoning}"))
        else:  # retry
            if downgraded_from_json:
                # The point of the retry is still JSON validity; harden the
                # prompt for format rather than the generic quality suffix.
                cfg = RunConfig(prompt_suffix=JSON_HARDENING_SUFFIX)
            else:
                cfg = _fix_to_config(fix)
            next_attempt = (run.attempt + 1) if run else 1
            pending.append(eval_payload(spec_for(state, d.model_id), w, next_attempt, cfg))
            log.append(narrate(
                f"supervise: RETRY {d.model_id} with fix '{fix}' (attempt {next_attempt}) — {d.reasoning}"
            ))

    # Any active candidate the LLM forgot to rule on: eliminate at cap, retry-less.
    for m in active:
        if m not in decided:
            statuses[m] = "eliminated" if iterations >= s.max_iterations else statuses[m]
            log.append(narrate(f"supervise: no ruling returned for {m} (left {'eliminated' if iterations >= s.max_iterations else 'active'})"))

    return {"statuses": statuses, "pending": pending, "iterations": iterations, "decision_log": log}
