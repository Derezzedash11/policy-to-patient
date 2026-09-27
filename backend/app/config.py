"""Runtime configuration read from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SAMPLE_DIR = REPO_ROOT / "data" / "sample"


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    return float(raw) if raw not in (None, "") else default


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    return int(raw) if raw not in (None, "") else default


@dataclass(frozen=True)
class Settings:
    data_dir: Path = field(
        default_factory=lambda: Path(os.environ.get("DATA_DIR") or REPO_ROOT / "data" / "runtime")
    )
    anthropic_api_key: str | None = field(
        default_factory=lambda: os.environ.get("ANTHROPIC_API_KEY") or None
    )
    llm_model: str = field(default_factory=lambda: os.environ.get("LLM_MODEL") or "claude-opus-5")
    min_evidence_score: float = field(
        default_factory=lambda: _env_float("MIN_EVIDENCE_SCORE", 0.08)
    )
    retrieval_top_k: int = field(default_factory=lambda: _env_int("RETRIEVAL_TOP_K", 5))
    max_upload_mb: int = field(default_factory=lambda: _env_int("MAX_UPLOAD_MB", 20))
    cost_table_path: Path = field(
        default_factory=lambda: Path(
            os.environ.get("COST_TABLE_PATH") or SAMPLE_DIR / "demo_treatment_costs.csv"
        )
    )
    cost_modifiers_path: Path = field(
        default_factory=lambda: Path(
            os.environ.get("COST_MODIFIERS_PATH") or SAMPLE_DIR / "demo_cost_modifiers.csv"
        )
    )
