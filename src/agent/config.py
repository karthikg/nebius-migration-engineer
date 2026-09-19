from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _load_env() -> None:
    for candidate in (Path.cwd() / ".env", PROJECT_ROOT / ".env"):
        if candidate.exists():
            load_dotenv(candidate, override=False)


@dataclass(frozen=True)
class Settings:
    api_key: str
    base_url: str
    supervisor_model: str
    judge_model: str
    default_candidate_pool: tuple[str, ...]
    max_iterations: int
    request_timeout_s: float
    pricing_file: Path
    runs_dir: Path


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    _load_env()
    api_key = os.getenv("NEBIUS_API_KEY") or os.getenv("TOKEN_FACTORY_API_KEY") or ""
    if not api_key:
        raise SystemExit(
            "NEBIUS_API_KEY is not set. Copy .env.example to .env and add your "
            "Token Factory API key."
        )
    pool = os.getenv(
        "CANDIDATE_POOL",
        "Qwen/Qwen3-30B-A3B-Instruct-2507,google/gemma-3-27b-it",
    )
    return Settings(
        api_key=api_key,
        base_url=os.getenv("NEBIUS_BASE_URL", "https://api.tokenfactory.nebius.com/v1/"),
        supervisor_model=os.getenv("SUPERVISOR_MODEL", "Qwen/Qwen3-235B-A22B-Instruct-2507"),
        judge_model=os.getenv("JUDGE_MODEL", "deepseek-ai/DeepSeek-V4-Pro"),
        default_candidate_pool=tuple(m.strip() for m in pool.split(",") if m.strip()),
        max_iterations=int(os.getenv("MAX_ITERATIONS", "2")),
        request_timeout_s=float(os.getenv("REQUEST_TIMEOUT_S", "120")),
        pricing_file=Path(os.getenv("PRICING_FILE", str(PROJECT_ROOT / "data" / "pricing.yaml"))),
        runs_dir=Path(os.getenv("RUNS_DIR", str(PROJECT_ROOT / "runs"))),
    )
