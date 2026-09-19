from __future__ import annotations

import operator
from typing import Annotated, Literal, Optional, TypedDict

from pydantic import BaseModel, Field


# --------------------------------------------------------------------------
# Workload spec (the agent's input)
# --------------------------------------------------------------------------

class VolumeProfile(BaseModel):
    requests_per_day: int
    avg_input_tokens: int
    avg_output_tokens: int


class BaselineModel(BaseModel):
    name: str
    input_price_per_mtok: float
    output_price_per_mtok: float


class Constraints(BaseModel):
    require_valid_json: bool = True
    required_keys: list[str] = Field(default_factory=list)
    max_p95_latency_s: Optional[float] = None
    min_cost_reduction_pct: Optional[float] = None
    min_quality_score: float = 4.0


class Sample(BaseModel):
    input: str
    reference_output: str


class WorkloadSpec(BaseModel):
    name: str
    task_description: str
    prompt_template: str
    max_output_tokens: int = 512
    volume: VolumeProfile
    baseline: BaselineModel
    constraints: Constraints
    samples: list[Sample]


# --------------------------------------------------------------------------
# Candidate evaluation
# --------------------------------------------------------------------------

class CandidateSpec(BaseModel):
    model_id: str
    reason: str = ""


class CandidatePlan(BaseModel):
    """Planner's structured pick of candidate models."""
    candidates: list[CandidateSpec]


class RunConfig(BaseModel):
    response_format_json: bool = False
    prompt_suffix: Optional[str] = None
    temperature: float = 0.0

    def describe(self) -> str:
        parts = []
        if self.response_format_json:
            parts.append("response_format=json_object")
        if self.prompt_suffix:
            parts.append("hardened prompt")
        return ", ".join(parts) if parts else "default config"


class SampleResult(BaseModel):
    sample_index: int
    output: str
    latency_s: float
    json_valid: bool
    missing_keys: list[str] = Field(default_factory=list)
    error: Optional[str] = None


class CandidateRun(BaseModel):
    model_id: str
    attempt: int
    config: RunConfig
    results: list[SampleResult]
    p95_latency_s: float
    json_valid_count: int
    n_samples: int


# --------------------------------------------------------------------------
# Judge
# --------------------------------------------------------------------------

class SampleScore(BaseModel):
    sample_index: int
    score: float = Field(ge=1, le=5, description="1=unusable, 5=equivalent to reference")
    notes: str = ""


class JudgeScores(BaseModel):
    """Raw structured output the judge model produces for one candidate run."""
    scores: list[SampleScore]
    summary: str


class JudgeVerdict(BaseModel):
    model_id: str
    attempt: int
    avg_score: float
    scores: list[SampleScore]
    summary: str


# --------------------------------------------------------------------------
# Costs
# --------------------------------------------------------------------------

class CostReport(BaseModel):
    model_id: str
    monthly_cost_usd: Optional[float]
    baseline_monthly_cost_usd: float
    savings_pct: Optional[float]
    meets_budget: Optional[bool]  # None = constraint not set or price unknown


# --------------------------------------------------------------------------
# Supervisor
# --------------------------------------------------------------------------

FixKind = Literal["enforce_json_response_format", "strengthen_prompt"]


class CandidateDecision(BaseModel):
    model_id: str
    action: Literal["accept", "eliminate", "retry"]
    fix: Optional[FixKind] = None
    reasoning: str


class SupervisorRuling(BaseModel):
    decisions: list[CandidateDecision]
    commentary: str


# --------------------------------------------------------------------------
# Final recommendation
# --------------------------------------------------------------------------

class CandidateSummaryRow(BaseModel):
    model_id: str
    attempt: int
    avg_quality: Optional[float]
    json_valid: str  # e.g. "5/5"
    p95_latency_s: float
    monthly_cost_usd: Optional[float]
    savings_pct: Optional[float]
    status: str


class ReportNarrative(BaseModel):
    confidence: Literal["high", "medium", "low"]
    summary: str
    migration_notes: str
    caveats: list[str]


class MigrationRecommendation(BaseModel):
    verdict: Literal["MIGRATE", "DONT_MIGRATE"]
    recommended_model: Optional[str]
    confidence: Literal["high", "medium", "low"]
    summary: str
    migration_notes: str
    caveats: list[str]
    rows: list[CandidateSummaryRow]


# --------------------------------------------------------------------------
# Graph state
# --------------------------------------------------------------------------

class GraphState(TypedDict, total=False):
    workload: WorkloadSpec
    raw_request: str
    candidates_override: list[str]
    candidates: list[CandidateSpec]
    statuses: dict[str, str]  # model_id -> active | accepted | eliminated
    pending: list[dict]       # evaluate payloads for the next fan-out
    iterations: int
    runs: Annotated[list[CandidateRun], operator.add]
    verdicts: Annotated[list[JudgeVerdict], operator.add]
    costs: list[CostReport]
    decision_log: Annotated[list[str], operator.add]
    recommendation: Optional[MigrationRecommendation]
    report_path: str
