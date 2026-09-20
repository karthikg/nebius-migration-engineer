"""Demo web UI backend: wraps the LangGraph agent behind a small FastAPI app.

Design notes for partners reading this as a reference:
- One run at a time (a demo server, not a job queue). POST /api/runs returns
  409 while a run is active.
- The UI polls GET /api/runs/{id}?after=N for incremental events instead of
  SSE/websockets — fewer moving parts, identical demo UX.
- Events are the graph's own node updates: every decision-log line the agent
  produces is streamed to the browser as it happens.
"""
from __future__ import annotations

import threading
import time
import uuid
from pathlib import Path
from typing import Optional

import yaml
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from .config import PROJECT_ROOT, get_settings
from .schemas import WorkloadSpec

app = FastAPI(title="Migration Engineer", docs_url=None, redoc_url=None)

_WEB_DIR = PROJECT_ROOT / "web"
_WORKLOADS_DIR = PROJECT_ROOT / "workloads"


class RunRecord:
    def __init__(self, rid: str, workload_name: str):
        self.id = rid
        self.workload = workload_name
        self.status = "running"  # running | done | error
        self.events: list[dict] = []
        self.final: Optional[dict] = None
        self.error: Optional[str] = None
        self._lock = threading.Lock()

    def add(self, evt: dict) -> None:
        with self._lock:
            self.events.append(evt)

    def slice(self, after: int) -> list[dict]:
        with self._lock:
            return self.events[after:]


RUNS: dict[str, RunRecord] = {}
_ACTIVE = threading.Lock()  # released by the worker thread when the run ends


class StartRequest(BaseModel):
    workload: str
    candidates: Optional[list[str]] = None


class PreflightRequest(BaseModel):
    models: Optional[list[str]] = None


def _execute(record: RunRecord, state: dict) -> None:
    from .graph import build_graph

    try:
        graph = build_graph()
        last_values: dict = {}
        for mode, chunk in graph.stream(
            state, config={"recursion_limit": 100}, stream_mode=["updates", "values"]
        ):
            if mode == "values":
                last_values = chunk
                continue
            for node, delta in (chunk or {}).items():
                record.add({
                    "type": "node",
                    "node": node,
                    "ts": time.time(),
                    "log": [str(x) for x in (delta or {}).get("decision_log", [])],
                })
        rec = last_values.get("recommendation")
        record.final = {
            "recommendation": rec.model_dump() if rec else None,
            "report_path": last_values.get("report_path"),
        }
        record.status = "done"
    except Exception as e:  # noqa: BLE001 - surfaced to the UI
        record.error = f"{type(e).__name__}: {e}"
        record.status = "error"
    finally:
        _ACTIVE.release()


@app.get("/api/workloads")
def list_workloads() -> list[dict]:
    out = []
    for p in sorted(_WORKLOADS_DIR.glob("*.yaml")):
        try:
            d = yaml.safe_load(p.read_text()) or {}
        except yaml.YAMLError:
            continue
        out.append({
            "file": p.name,
            "name": d.get("name", p.stem),
            "task": " ".join((d.get("task_description") or "").split())[:220],
            "baseline": (d.get("baseline") or {}).get("name", "?"),
            "samples": len(d.get("samples") or []),
        })
    return out


@app.get("/api/models")
def list_models() -> list[dict]:
    from .tools import fetch_catalog

    return [
        {
            "id": m["id"],
            "features": m.get("supported_features", []),
            "context_length": m.get("context_length"),
        }
        for m in sorted(fetch_catalog(), key=lambda m: m["id"])
    ]


@app.post("/api/runs")
def start_run(req: StartRequest) -> dict:
    path = _WORKLOADS_DIR / Path(req.workload).name
    if not path.exists():
        raise HTTPException(404, f"unknown workload: {req.workload}")
    if not _ACTIVE.acquire(blocking=False):
        raise HTTPException(409, "a run is already in progress")
    try:
        spec = WorkloadSpec.model_validate(yaml.safe_load(path.read_text()))
        state: dict = {"workload": spec}
        if req.candidates:
            state["candidates_override"] = req.candidates
        rid = uuid.uuid4().hex[:8]
        record = RunRecord(rid, spec.name)
        RUNS[rid] = record
    except Exception:
        _ACTIVE.release()
        raise
    threading.Thread(target=_execute, args=(record, state), daemon=True).start()
    return {"run_id": rid, "workload": spec.name}


@app.get("/api/runs/{rid}")
def poll_run(rid: str, after: int = 0) -> dict:
    record = RUNS.get(rid)
    if not record:
        raise HTTPException(404, "unknown run")
    events = record.slice(after)
    return {
        "status": record.status,
        "events": events,
        "next": after + len(events),
        "final": record.final if record.status == "done" else None,
        "error": record.error,
    }


@app.post("/api/preflight")
def preflight(req: PreflightRequest) -> dict:
    from .preflight import run_preflight

    s = get_settings()
    models = req.models or [s.supervisor_model, s.judge_model, *s.default_candidate_pool]
    models = list(dict.fromkeys(models))[:6]
    results = run_preflight(models)
    return {
        "models": [
            {
                "id": m,
                "checks": {name: {"ok": ok, "note": note} for name, (ok, note) in checks.items()},
            }
            for m, checks in results.items()
        ]
    }


@app.get("/")
def index() -> FileResponse:
    return FileResponse(_WEB_DIR / "index.html")


def main(host: str = "127.0.0.1", port: int = 8765) -> None:
    import uvicorn

    uvicorn.run(app, host=host, port=port, log_level="warning")
