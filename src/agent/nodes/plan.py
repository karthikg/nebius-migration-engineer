from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

from ..config import get_settings
from ..llm import chat_model, structured_completion
from ..logging_utils import narrate
from ..prompts import PLANNER_PICK_SYSTEM, PLANNER_SYSTEM
from ..schemas import CandidatePlan, CandidateSpec, GraphState
from ..tools import catalog_ids, get_pricing, list_models
from .common import eval_payload

N_CANDIDATES = 2
MAX_TOOL_ROUNDS = 4


def _fallback_candidates(available: set[str], log: list[str]) -> list[CandidateSpec]:
    s = get_settings()
    picks = [m for m in s.default_candidate_pool if m in available][:N_CANDIDATES]
    log.append(narrate(f"plan: falling back to configured candidate pool: {picks}"))
    return [CandidateSpec(model_id=m, reason="from configured fallback pool") for m in picks]


def plan(state: GraphState) -> dict:
    """Supervisor picks candidate models from the LIVE Token Factory catalog.

    This is the node that exercises tool calling through LangGraph/LangChain
    abstractions: the planner LLM calls list_models / get_pricing and we
    execute them and feed results back until it commits to a pick.
    """
    s = get_settings()
    w = state["workload"]
    log: list[str] = []
    available = catalog_ids()

    # User-pinned candidates skip the LLM planner entirely (reproducible demos).
    if state.get("candidates_override"):
        picks = []
        for m in state["candidates_override"]:
            if m in available:
                picks.append(CandidateSpec(model_id=m, reason="pinned by user via --candidates"))
            else:
                log.append(narrate(f"plan: pinned candidate {m} not in live catalog — skipping"))
        if not picks:
            raise ValueError("None of the pinned --candidates exist in the live catalog.")
        log.append(narrate(f"plan: using pinned candidates: {[c.model_id for c in picks]}"))
    else:
        llm = chat_model(s.supervisor_model, tags=["planner"]).bind_tools([list_models, get_pricing])
        tool_fns = {"list_models": list_models, "get_pricing": get_pricing}
        messages: list = [
            SystemMessage(PLANNER_SYSTEM.format(n_candidates=N_CANDIDATES)),
            HumanMessage(
                f"Workload: {w.name}\n"
                f"Task: {w.task_description}\n"
                f"Volume: {w.volume.requests_per_day}/day, "
                f"~{w.volume.avg_input_tokens} in / {w.volume.avg_output_tokens} out tokens\n"
                f"Current model: {w.baseline.name} "
                f"(${w.baseline.input_price_per_mtok}/M in, ${w.baseline.output_price_per_mtok}/M out)\n"
                f"Constraints: {w.constraints.model_dump()}\n\n"
                f"Inspect the catalog and choose {N_CANDIDATES} candidates."
            ),
        ]
        for _ in range(MAX_TOOL_ROUNDS):
            ai = llm.invoke(messages)
            messages.append(ai)
            if not ai.tool_calls:
                break
            for tc in ai.tool_calls:
                fn = tool_fns.get(tc["name"])
                out = fn.invoke(tc["args"]) if fn else f"unknown tool {tc['name']}"
                log.append(narrate(f"plan: tool {tc['name']}({tc['args']}) -> {str(out)[:100]}"))
                messages.append(ToolMessage(content=str(out), tool_call_id=tc["id"]))

        # Extract structured picks from the whole planning transcript — the
        # tool results carry the catalog even if the model never wrote prose.
        transcript = "\n".join(
            f"[{m.type}] {str(m.content)[:800]}" for m in messages[1:] if str(m.content).strip()
        )
        picks: list[CandidateSpec] = []
        try:
            plan_out = structured_completion(
                s.supervisor_model, CandidatePlan, PLANNER_PICK_SYSTEM, transcript
            )
            picks = [c for c in plan_out.candidates if c.model_id in available][:N_CANDIDATES]
            for c in plan_out.candidates:
                if c.model_id not in available:
                    log.append(narrate(f"plan: planner picked {c.model_id} which is NOT in the live catalog — rejected"))
        except Exception as e:  # noqa: BLE001
            log.append(narrate(f"plan: could not extract structured picks ({type(e).__name__})"))
        if not picks:
            picks = _fallback_candidates(available, log)
        for c in picks:
            log.append(narrate(f"plan: candidate {c.model_id} — {c.reason}"))

    statuses = {c.model_id: "active" for c in picks}
    pending = [eval_payload(c, w, attempt=1) for c in picks]
    return {"candidates": picks, "statuses": statuses, "pending": pending, "decision_log": log}
