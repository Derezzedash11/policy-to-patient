"""Local file storage for ingested policies (Phase 2: PostgreSQL behind the same methods)."""

from __future__ import annotations

import json
from pathlib import Path

from app.models import Chunk, PageText, PolicySummary


class PolicyStore:
    def __init__(self, root: Path) -> None:
        self._root = root / "policies"
        self._root.mkdir(parents=True, exist_ok=True)

    def _dir(self, doc_id: str) -> Path:
        if not doc_id.isalnum():
            raise KeyError(doc_id)
        return self._root / doc_id

    def save(self, summary: PolicySummary, pdf: bytes, pages: list[PageText], chunks: list[Chunk]) -> None:
        d = self._dir(summary.doc_id)
        d.mkdir(parents=True, exist_ok=True)
        (d / "source.pdf").write_bytes(pdf)
        (d / "pages.json").write_text(json.dumps([p.model_dump() for p in pages], indent=1))
        (d / "chunks.json").write_text(json.dumps([c.model_dump() for c in chunks], indent=1))
        (d / "summary.json").write_text(summary.model_dump_json(indent=1))

    def exists(self, doc_id: str) -> bool:
        try:
            return (self._dir(doc_id) / "summary.json").is_file()
        except KeyError:
            return False

    def get_summary(self, doc_id: str) -> PolicySummary:
        if not self.exists(doc_id):
            raise KeyError(doc_id)
        return PolicySummary.model_validate_json((self._dir(doc_id) / "summary.json").read_text())

    def get_chunks(self, doc_id: str) -> list[Chunk]:
        if not self.exists(doc_id):
            raise KeyError(doc_id)
        raw = json.loads((self._dir(doc_id) / "chunks.json").read_text())
        return [Chunk.model_validate(c) for c in raw]

    def list_summaries(self) -> list[PolicySummary]:
        return [
            PolicySummary.model_validate_json(p.read_text())
            for p in sorted(self._root.glob("*/summary.json"))
        ]
