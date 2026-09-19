from __future__ import annotations

import json
import re
from typing import Any, Optional, Type, TypeVar

from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from .config import get_settings

T = TypeVar("T", bound=BaseModel)


def chat_model(
    model_id: str,
    *,
    temperature: float = 0.0,
    max_tokens: Optional[int] = None,
    json_mode: bool = False,
    tags: Optional[list[str]] = None,
) -> Any:
    """A ChatOpenAI instance pointed at Nebius Token Factory."""
    s = get_settings()
    llm = ChatOpenAI(
        model=model_id,
        api_key=s.api_key,
        base_url=s.base_url,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=s.request_timeout_s,
        max_retries=2,
        tags=tags or [],
    )
    if json_mode:
        llm = llm.bind(response_format={"type": "json_object"})
    return llm


_FENCE_RE = re.compile(r"^```[a-zA-Z]*\s*|\s*```$", re.MULTILINE)


def extract_json(text: str) -> Any:
    """Best-effort extraction of a JSON object from model text."""
    t = text.strip()
    try:
        return json.loads(t)
    except (json.JSONDecodeError, ValueError):
        pass
    t2 = _FENCE_RE.sub("", t).strip()
    try:
        return json.loads(t2)
    except (json.JSONDecodeError, ValueError):
        pass
    m = re.search(r"\{.*\}", t, re.DOTALL)
    if m:
        return json.loads(m.group(0))
    raise ValueError(f"No JSON object found in model output: {text[:200]!r}")


def structured_completion(
    model_id: str,
    schema: Type[T],
    system: str,
    user: str,
) -> T:
    """Structured output with graceful degradation.

    Tries the OpenAI-compat json_schema response format first (vLLM guided
    decoding on Token Factory), then json_object mode with the schema in the
    prompt, then plain text + JSON extraction. Which path works per model is
    exactly what `tf-migrate preflight` measures.
    """
    base = chat_model(model_id)
    plain = [("system", system), ("user", user)]
    schema_hint = (
        "Respond ONLY with a single JSON object that conforms to this JSON Schema "
        "(no markdown fences, no prose):\n" + json.dumps(schema.model_json_schema())
    )
    hinted = [("system", system + "\n\n" + schema_hint), ("user", user)]

    errors: list[str] = []
    try:
        return base.with_structured_output(schema, method="json_schema").invoke(plain)
    except Exception as e:  # noqa: BLE001 - fall through to the next strategy
        errors.append(f"json_schema: {type(e).__name__}: {e}")
    try:
        return base.with_structured_output(schema, method="json_mode").invoke(hinted)
    except Exception as e:  # noqa: BLE001
        errors.append(f"json_mode: {type(e).__name__}: {e}")
    try:
        raw = base.invoke(hinted).content
        return schema.model_validate(extract_json(raw))
    except Exception as e:  # noqa: BLE001
        errors.append(f"extraction: {type(e).__name__}: {e}")
    raise RuntimeError(
        f"All structured-output strategies failed for {model_id} -> "
        f"{schema.__name__}:\n" + "\n".join(errors)
    )
