from __future__ import annotations

import json
from typing import Any, Optional

import httpx
import yaml
from langchain_core.tools import tool

from .config import get_settings

_catalog_cache: Optional[list[dict]] = None


def fetch_catalog(force: bool = False) -> list[dict]:
    """The live model catalog from Token Factory's /models endpoint.

    verbose=true adds per-model pricing, context length and supported_features
    (tools / json_mode / structured_outputs / reasoning) — the agent reasons
    over these instead of assuming capabilities.
    """
    global _catalog_cache
    if _catalog_cache is None or force:
        s = get_settings()
        resp = httpx.get(
            s.base_url.rstrip("/") + "/models",
            params={"verbose": "true"},
            headers={"Authorization": f"Bearer {s.api_key}"},
            timeout=30,
        )
        resp.raise_for_status()
        _catalog_cache = resp.json().get("data", [])
    return _catalog_cache


def catalog_ids() -> set[str]:
    return {m["id"] for m in fetch_catalog()}


def model_info(model_id: str) -> Optional[dict]:
    for m in fetch_catalog():
        if m["id"] == model_id:
            return m
    return None


def model_features(model_id: str) -> list[str]:
    m = model_info(model_id)
    return list(m.get("supported_features", [])) if m else []


def live_pricing() -> dict[str, dict]:
    """USD per 1M tokens straight from the live catalog (authoritative)."""
    table: dict[str, dict] = {}
    for m in fetch_catalog():
        p = m.get("pricing") or {}
        try:
            inp = float(p.get("prompt", 0)) * 1_000_000
            out = float(p.get("completion", 0)) * 1_000_000
        except (TypeError, ValueError):
            continue
        if inp <= 0 and out <= 0:
            continue
        table[m["id"]] = {
            "input_price_per_mtok": round(inp, 4),
            "output_price_per_mtok": round(out, 4),
            "source": "live catalog",
        }
    return table


def pricing_table() -> dict[str, dict]:
    """Live API pricing first; data/pricing.yaml as offline fallback."""
    s = get_settings()
    table: dict[str, dict] = {}
    if s.pricing_file.exists():
        with open(s.pricing_file) as f:
            data = yaml.safe_load(f) or {}
        for mid, p in (data.get("models") or {}).items():
            table[mid] = {**p, "source": "fallback table"}
    try:
        table.update(live_pricing())
    except httpx.HTTPError:
        pass  # offline: fallback table stands
    return table


@tool
def list_models(keyword: str = "") -> str:
    """List models available on Nebius Token Factory right now with their
    supported features (tools, json_mode, structured_outputs, reasoning) and
    context length. Optional case-insensitive keyword filter (e.g. 'qwen')."""
    rows = [
        {
            "id": m["id"],
            "features": m.get("supported_features", []),
            "context_length": m.get("context_length"),
        }
        for m in fetch_catalog()
    ]
    if keyword:
        rows = [r for r in rows if keyword.lower() in r["id"].lower()]
    return json.dumps(sorted(rows, key=lambda r: r["id"]))


@tool
def get_pricing(model_id: str) -> str:
    """Get input/output prices in USD per 1M tokens for a Token Factory model id
    (live catalog pricing, with a local fallback table)."""
    p = pricing_table().get(model_id)
    if not p:
        return json.dumps({"model": model_id, "pricing": "unknown"})
    return json.dumps({"model": model_id, **p})


_json_probe_cache: dict[str, bool] = {}


def probe_json_mode(model_id: str) -> bool:
    """Live-probe whether this model actually honors response_format
    json_object on the endpoint. Published metadata and real behavior can
    disagree in both directions — when a decision depends on it, ask the
    endpoint, not the catalog."""
    if model_id not in _json_probe_cache:
        s = get_settings()
        try:
            r = httpx.post(
                s.base_url.rstrip("/") + "/chat/completions",
                headers={"Authorization": f"Bearer {s.api_key}"},
                json={
                    "model": model_id,
                    "max_tokens": 60,
                    "response_format": {"type": "json_object"},
                    "messages": [{"role": "user", "content": 'Return a JSON object: {"ok": true}'}],
                },
                timeout=30,
            )
            r.raise_for_status()
            content = r.json()["choices"][0]["message"]["content"]
            json.loads(content)
            _json_probe_cache[model_id] = True
        except Exception:  # noqa: BLE001 - any failure means "can't rely on it"
            _json_probe_cache[model_id] = False
    return _json_probe_cache[model_id]


def parse_json_strict(text: str) -> tuple[Optional[Any], Optional[str]]:
    """Strict production-style parse: the raw response must BE the JSON object.

    Markdown fences or surrounding prose fail here on purpose — that is what a
    downstream pipeline consuming the response would experience.
    """
    try:
        return json.loads(text.strip()), None
    except (json.JSONDecodeError, ValueError) as e:
        return None, str(e)


def validate_output(text: str, required_keys: list[str]) -> tuple[bool, list[str], Optional[str]]:
    """Returns (valid, missing_keys, parse_error)."""
    data, err = parse_json_strict(text)
    if data is None or not isinstance(data, dict):
        return False, list(required_keys), err or "not a JSON object"
    missing = [k for k in required_keys if k not in data]
    return len(missing) == 0, missing, None
