"""Capability preflight for Token Factory models.

"Verify that tool calling actually works through the abstractions; don't
assume it does." — this module is that verification, at the raw OpenAI-compat
layer. Run it against every model you plan to give a role to (supervisor,
judge, candidates) BEFORE trusting a run. It is also the reusable asset for
the next framework integration: same checks, any OpenAI-compatible endpoint.
"""
from __future__ import annotations

import json

from openai import OpenAI
from rich.console import Console
from rich.table import Table

from .config import get_settings

console = Console()

_WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}

_OK_SCHEMA = {
    "name": "ok_check",
    "schema": {
        "type": "object",
        "properties": {"ok": {"type": "boolean"}},
        "required": ["ok"],
        "additionalProperties": False,
    },
}


def _client() -> OpenAI:
    s = get_settings()
    return OpenAI(api_key=s.api_key, base_url=s.base_url, timeout=60, max_retries=1)


def check_basic(client: OpenAI, model: str):
    r = client.chat.completions.create(
        model=model, max_tokens=10,
        messages=[{"role": "user", "content": "Reply with the single word: OK"}],
    )
    text = r.choices[0].message.content or ""
    return bool(text.strip()), text.strip()[:40]


def check_tool_call(client: OpenAI, model: str):
    r = client.chat.completions.create(
        model=model, max_tokens=200, tools=[_WEATHER_TOOL],
        messages=[{"role": "user", "content": "What's the weather in Berlin? Use the tool."}],
    )
    tcs = r.choices[0].message.tool_calls
    if not tcs:
        return False, "no tool_calls in response"
    args = json.loads(tcs[0].function.arguments)
    return "city" in args, f"called {tcs[0].function.name}({args})"


def check_forced_tool(client: OpenAI, model: str):
    r = client.chat.completions.create(
        model=model, max_tokens=200, tools=[_WEATHER_TOOL],
        tool_choice={"type": "function", "function": {"name": "get_weather"}},
        messages=[{"role": "user", "content": "Hello there."}],
    )
    tcs = r.choices[0].message.tool_calls
    return bool(tcs), "forced call honored" if tcs else "tool_choice ignored"


def check_json_object(client: OpenAI, model: str):
    r = client.chat.completions.create(
        model=model, max_tokens=100,
        response_format={"type": "json_object"},
        messages=[{"role": "user", "content": 'Return a JSON object: {"ok": true}'}],
    )
    data = json.loads(r.choices[0].message.content)
    return isinstance(data, dict), "parsed"


def check_json_schema(client: OpenAI, model: str):
    r = client.chat.completions.create(
        model=model, max_tokens=100,
        response_format={"type": "json_schema", "json_schema": _OK_SCHEMA},
        messages=[{"role": "user", "content": "Return ok=true."}],
    )
    data = json.loads(r.choices[0].message.content)
    return isinstance(data, dict) and "ok" in data, "schema respected"


def check_streaming(client: OpenAI, model: str):
    chunks = 0
    text = ""
    for chunk in client.chat.completions.create(
        model=model, max_tokens=50, stream=True,
        messages=[{"role": "user", "content": "Count from 1 to 5, digits only."}],
    ):
        chunks += 1
        if chunk.choices and chunk.choices[0].delta.content:
            text += chunk.choices[0].delta.content
    return chunks > 1 and bool(text.strip()), f"{chunks} chunks"


CHECKS = [
    ("basic", check_basic),
    ("tool_call", check_tool_call),
    ("forced_tool", check_forced_tool),
    ("json_object", check_json_object),
    ("json_schema", check_json_schema),
    ("streaming", check_streaming),
]


def run_preflight(models: list[str]) -> dict[str, dict[str, tuple[bool, str]]]:
    client = _client()
    results: dict[str, dict[str, tuple[bool, str]]] = {}
    for model in models:
        console.print(f"[bold]preflight:[/bold] {model}")
        results[model] = {}
        for name, fn in CHECKS:
            try:
                ok, note = fn(client, model)
            except Exception as e:  # noqa: BLE001 - a failed check is a result
                ok, note = False, f"{type(e).__name__}: {str(e)[:80]}"
            results[model][name] = (ok, note)
            style = "green" if ok else "red"
            console.print(f"  [{style}]{'✓' if ok else '✗'}[/{style}] {name}: {note}")
    return results


def render_table(results: dict[str, dict[str, tuple[bool, str]]]) -> Table:
    table = Table(title="Token Factory capability matrix")
    table.add_column("model", style="bold")
    for name, _ in CHECKS:
        table.add_column(name, justify="center")
    for model, checks in results.items():
        row = [model]
        for name, _ in CHECKS:
            ok, _note = checks.get(name, (False, ""))
            row.append("[green]✓[/green]" if ok else "[red]✗[/red]")
        table.add_row(*row)
    return table


def markdown_matrix(results: dict[str, dict[str, tuple[bool, str]]]) -> str:
    header = "| model | " + " | ".join(name for name, _ in CHECKS) + " |"
    sep = "|---" * (len(CHECKS) + 1) + "|"
    lines = [header, sep]
    for model, checks in results.items():
        cells = ["✅" if checks.get(name, (False, ""))[0] else "❌" for name, _ in CHECKS]
        lines.append(f"| {model} | " + " | ".join(cells) + " |")
    return "\n".join(lines)
