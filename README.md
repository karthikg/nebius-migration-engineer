# Migration Engineer — a LangGraph multi-agent on Nebius Token Factory

**Give it a workload you run on a closed model. It finds, evaluates, and
validates the best open model on [Nebius Token Factory](https://nebius.com)
to migrate to — with evidence.**

A supervisor agent plans candidate models from the **live** Token Factory
catalog, fans out parallel evaluation sub-agents, has an LLM judge score the
outputs against your current model's references, runs a deterministic cost
analysis, and then *reasons about the evidence*: accept, eliminate, or **retry
a candidate with a concrete fix** (e.g. enforce `response_format=json_object`)
before issuing a final `MIGRATE` / `DONT_MIGRATE` recommendation.

Every LLM call in the system — supervisor, evaluators, judge, report writer —
runs on Token Factory's OpenAI-compatible API.

```
intake → plan ──► evaluate (parallel, one per candidate) ──► judge ──► costs ──► supervise
                      ▲                                                             │
                      └── retry with fix (bounded) ─────────────────────────────────┤
                                                                                    ▼
                                                                                 report
```

## Quickstart (5 minutes)

Requires Python 3.11+ and a Token Factory API key.

```bash
git clone https://github.com/<you>/nebius-migration-engineer.git
cd nebius-migration-engineer
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
cp .env.example .env    # then set NEBIUS_API_KEY
```

**Step 1 — verify the platform, don't assume it** (this is the habit this repo
is built around):

```bash
tf-migrate preflight
```

This probes every model the agent will use (supervisor, judge, candidate pool)
for the capabilities the agent depends on — tool calling, forced tool choice,
`json_object` / `json_schema` response formats, streaming — at the raw
OpenAI-compat layer:

| model | basic | tool_call | forced_tool | json_object | json_schema | streaming |
|---|---|---|---|---|---|---|
| Qwen/Qwen3-235B-A22B-Instruct-2507 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| deepseek-ai/DeepSeek-V4-Pro | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Qwen/Qwen3-30B-A3B-Instruct-2507 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| google/gemma-3-27b-it | ✅ | ❌ | ✅ | ✅ | ✅ | ✅ |

*(measured live on 2026-09-19 — note gemma: no voluntary tool calls, but
`response_format` works even though the catalog metadata doesn't advertise
`json_mode`. Metadata and behavior disagree in both directions, which is
exactly why this harness exists.)*

**Step 2 — run a migration evaluation:**

```bash
tf-migrate run workloads/summarization.yaml
```

You'll watch the decision log stream as the graph executes, then get a
verdict, an evidence table, and a full report under `runs/<timestamp>/`.

**Step 3 — or demo it in the browser:**

```bash
tf-migrate serve
```

opens a local web UI at `http://127.0.0.1:8765`: pick a workload, optionally
pin candidates from the live catalog, and watch the agent's decision log
stream node-by-node — plan tool calls, per-candidate evaluations, judge
scores, supervisor rulings, retries — ending in the verdict panel and
evidence table. There's also a one-click preflight matrix. The page is a
single dependency-free HTML file ([web/index.html](web/index.html)) served by
a small FastAPI app ([src/agent/server.py](src/agent/server.py)) that streams
the LangGraph run's own node updates; to demo from a Nebius VM, run it there
and tunnel: `ssh -L 8765:localhost:8765 ubuntu@<vm-ip>`.

Other entry points:

```bash
tf-migrate models                 # live Token Factory catalog
tf-migrate run workloads/extraction.yaml
tf-migrate run --prompt "I run a ticket-classification service on gpt-4o ..."   # free-text intake
tf-migrate run workloads/summarization.yaml --candidates "meta-llama/Llama-3.3-70B-Instruct,Qwen/Qwen2.5-72B-Instruct"
```

## Bring your own workload

Copy a file in [`workloads/`](workloads/) and edit. A workload is:

| Field | What it is |
|---|---|
| `task_description` | What the workload does; the planner and judge read this |
| `prompt_template` | Your actual production prompt, with `{input}` |
| `samples` | 3–5 real inputs + the outputs your **current** model produces |
| `volume` | requests/day and average token counts — drives the cost model |
| `baseline` | Your current model and its prices |
| `constraints` | JSON validity, required keys, p95 latency, cost-reduction target, minimum quality |

The reference outputs are what makes the evaluation real: the judge scores
candidates *against your current model's behavior*, not against vibes.

## What the agent actually reasons about

The supervisor's rulings are the interesting part. On each round it sees, per
candidate: JSON validity (checked **strictly** in code — a markdown-fenced
response fails, because that's what your downstream parser would experience),
judge quality scores vs. your references, measured p95 latency, and projected
monthly cost vs. your baseline. It then:

- **accepts** candidates that meet every constraint,
- **eliminates** candidates with content-level quality failures, and
- **retries** candidates whose failures look *fixable*, choosing the fix —
  turning on `response_format={"type": "json_object"}` for format failures, or
  hardening the prompt for marginal quality — with a bounded retry budget and
  a guardrail that never re-applies the same fix twice. When catalog metadata
  says a model lacks `json_mode` but the fix needs it, the agent **live-probes
  the endpoint** and trusts the probe over the metadata (they disagree in
  practice — run `tf-migrate preflight` and compare).

The full decision log lands in the report and in the trace, so you can audit
every ruling.

## Architecture

- **LangGraph** `StateGraph` with a typed state ([src/agent/schemas.py](src/agent/schemas.py)),
  parallel candidate evaluation via `Send` fan-out, and conditional edges for
  the retry loop ([src/agent/graph.py](src/agent/graph.py)).
- **Tool calling through the framework**: the planner node binds
  `list_models` / `get_pricing` LangChain tools and calls them against the
  live catalog ([src/agent/nodes/plan.py](src/agent/nodes/plan.py)).
- **Structured output, verified**: every structured call degrades gracefully
  `json_schema → json_object → text+extraction` ([src/agent/llm.py](src/agent/llm.py)),
  and `preflight` tells you which rungs a model actually supports.
- **Deterministic where it matters**: JSON validation, cost math, winner
  selection, and retry budgets are code, not vibes. LLMs plan, judge quality,
  and write narrative.
- **All prompts in one file** ([src/agent/prompts.py](src/agent/prompts.py)).

## Example runs (real, against the live endpoint)

Two curated evidence reports are committed in [`examples/`](examples/):

- [`report-summarization-retry-eliminate.md`](examples/report-summarization-retry-eliminate.md) —
  one candidate accepted outright; the other fails strict JSON, gets a retry
  with a hardened prompt, still fails, and is **eliminated** with the ruling
  citing the endpoint's capability flags.
- [`report-extraction-probe-fix.md`](examples/report-extraction-probe-fix.md) —
  gemma fails JSON 0/4 with good content; the supervisor orders the
  `response_format` fix, the **live probe overrides the catalog metadata**
  (which claims no `json_mode`), attempt 2 passes 4/4, and gemma **wins the
  recommendation** — with the required config carried into the migration notes.

## Tracing

Two layers, pick your depth:

1. **Always on**: a live decision log streams to the console and is persisted
   with full evidence (every sample output, judge score, and ruling) under
   `runs/<timestamp>/` (`report.md` + `state.json`).
2. **Full graph trace**: set `LANGSMITH_TRACING=true`, `LANGSMITH_API_KEY`
   and `LANGSMITH_PROJECT` in `.env` (free LangSmith account works). You get
   the complete LangGraph execution tree — the parallel evaluate fan-out and
   the retry edge firing are clearly visible. Nothing else in the repo changes.

## Running on Nebius Compute

The agent is a thin orchestrator — inference happens on Token Factory — so a
small CPU VM is enough: see [docs/deploy-nebius-vm.md](docs/deploy-nebius-vm.md)
and [scripts/bootstrap_vm.sh](scripts/bootstrap_vm.sh).

## For the next partner integration

Two pieces here are deliberately framework-agnostic:

- [`src/agent/preflight.py`](src/agent/preflight.py) / [`scripts/preflight.py`](scripts/preflight.py):
  the capability matrix runs against any OpenAI-compatible endpoint and any
  model — it's step 1 of onboarding *any* agent framework onto Token Factory.
- The graceful-degradation pattern in [`src/agent/llm.py`](src/agent/llm.py)
  for structured output, which encodes what open-model endpoints actually
  support rather than what SDKs assume.

Cost figures come from the **live catalog** (`GET /v1/models?verbose=true`
exposes per-model pricing, context length, and supported features like
`json_mode` — the agent reasons over these instead of assuming capabilities).
`data/pricing.yaml` is only an offline fallback snapshot.

## License

MIT
