#!/usr/bin/env python3
"""Standalone entry point: python scripts/preflight.py [model ...]

Equivalent to `tf-migrate preflight -m ...` — kept as a plain script so a
partner can lift preflight into any repo without installing this package.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agent.config import get_settings  # noqa: E402
from agent.preflight import markdown_matrix, render_table, run_preflight  # noqa: E402
from rich.console import Console  # noqa: E402

if __name__ == "__main__":
    s = get_settings()
    models = sys.argv[1:] or [s.supervisor_model, s.judge_model, *s.default_candidate_pool]
    models = list(dict.fromkeys(models))
    results = run_preflight(models)
    console = Console()
    console.print()
    console.print(render_table(results))
    print()
    print(markdown_matrix(results))
