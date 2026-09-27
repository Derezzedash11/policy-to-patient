"""Runtime configuration read from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SAMPLE_DIR = REPO_ROOT / "data" / "sample"


def _env_optional_float(name: str) -> float | None:
    raw = os.environ.get(name)
    return float(raw) if raw not in (None, "") else None


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
    # None → use the embedder's own default threshold (scores are not comparable across models).
    min_evidence_score: float | None = field(
        default_factory=lambda: _env_optional_float("MIN_EVIDENCE_SCORE")
    )
    embedding_provider: str = field(
        default_factory=lambda: os.environ.get("EMBEDDING_PROVIDER") or "fastembed"
    )
    embedding_model: str = field(
        default_factory=lambda: os.environ.get("EMBEDDING_MODEL") or "BAAI/bge-small-en-v1.5"
    )
    # PostgreSQL + pgvector when set; otherwise vectors are stored as local files under data_dir.
    database_url: str | None = field(default_factory=lambda: os.environ.get("DATABASE_URL") or None)
    retrieval_top_k: int = field(default_factory=lambda: _env_int("RETRIEVAL_TOP_K", 5))
    # Self-correcting retrieval: LLM query rewrites when no passage passes the threshold (0 = off).
    max_query_rewrites: int = field(default_factory=lambda: _env_int("MAX_QUERY_REWRITES", 1))
    max_upload_mb: int = field(default_factory=lambda: _env_int("MAX_UPLOAD_MB", 20))
    # Built UI (`npm run build` in frontend/); served at /ui/ when present.
    frontend_dist: Path = field(
        default_factory=lambda: Path(os.environ.get("FRONTEND_DIST") or REPO_ROOT / "frontend" / "dist")
    )
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
