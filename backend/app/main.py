"""FastAPI application factory. Run with: uvicorn app.main:create_app --factory"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes import router
from app.config import Settings
from app.cost.base import CostEstimator
from app.cost.demo_table import DemoTableEstimator
from app.embeddings import build_embedder
from app.embeddings.base import Embedder
from app.qa import build_llm
from app.qa.llm import LLMClient
from app.retrieval.vector import VectorRetriever
from app.services import PolicyService
from app.store import PolicyStore
from app.vectorstore import build_vector_store
from app.vectorstore.base import VectorStore

_UNSET = object()


def create_app(
    settings: Settings | None = None,
    llm: LLMClient | None | object = _UNSET,
    cost_estimator: CostEstimator | None = None,
    embedder: Embedder | None = None,
    vector_store: VectorStore | None = None,
) -> FastAPI:
    settings = settings or Settings()
    if llm is _UNSET:
        llm = build_llm(settings)
    app = FastAPI(
        title="Policy-to-Patient API (prototype)",
        description=(
            "Policy PDF → page-aware chunks → embeddings → vector search → grounded Q&A with "
            "citations → synthetic cost estimate → deterministic coverage/OOP. "
            "Prototype only: outputs are estimates, not "
            "insurance decisions, and cost figures are synthetic."
        ),
        version="0.3.0",
    )
    retriever = VectorRetriever(
        embedder or build_embedder(settings), vector_store or build_vector_store(settings)
    )
    app.state.policy_service = PolicyService(
        settings, PolicyStore(settings.data_dir), llm, retriever  # type: ignore[arg-type]
    )
    app.state.cost_estimator = cost_estimator or DemoTableEstimator(
        settings.cost_table_path, settings.cost_modifiers_path
    )
    app.include_router(router)
    if (settings.frontend_dist / "index.html").is_file():
        app.mount("/ui", StaticFiles(directory=settings.frontend_dist, html=True), name="ui")

        @app.get("/", include_in_schema=False)
        def root() -> RedirectResponse:
            return RedirectResponse("/ui/")

    return app
