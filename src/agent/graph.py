from __future__ import annotations

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from .nodes import costs, evaluate, intake, judge, plan, report, supervise
from .schemas import GraphState


def _fan_out(state: GraphState):
    """After plan/supervise: dispatch one parallel evaluate task per pending
    candidate payload (LangGraph map-reduce via Send)."""
    return [Send("evaluate", p) for p in state.get("pending", [])]


def _after_supervise(state: GraphState):
    if state.get("pending"):
        return _fan_out(state)
    return "report"


def build_graph():
    g = StateGraph(GraphState)
    g.add_node("intake", intake)
    g.add_node("plan", plan)
    g.add_node("evaluate", evaluate)
    g.add_node("judge", judge)
    g.add_node("costs", costs)
    g.add_node("supervise", supervise)
    g.add_node("report", report)

    g.add_edge(START, "intake")
    g.add_edge("intake", "plan")
    g.add_conditional_edges("plan", _fan_out, ["evaluate"])
    g.add_edge("evaluate", "judge")   # fan-in: judge runs once per round
    g.add_edge("judge", "costs")
    g.add_edge("costs", "supervise")
    g.add_conditional_edges("supervise", _after_supervise, ["evaluate", "report"])
    g.add_edge("report", END)
    return g.compile()
