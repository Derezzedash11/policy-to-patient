"""FastAPI application factory. Run with: uvicorn app.main:app --reload"""

from __future__ import annotations

from fastapi import FastAPI

from app.api.routes import router
from app.config import Settings
from app.cost.base import CostEstimator
from app.cost.demo_table import DemoTableEstimator
from app.qa.llm import AnthropicLLM, LLMClient
from app.services import PolicyService
from app.store import PolicyStore

_UNSET = object()


def create_app(
    settings: Settings | None = None,
    llm: LLMClient | None | object = _UNSET,
    cost_estimator: CostEstimator | None = None,
) -> FastAPI:
    settings = settings or Settings()
    if llm is _UNSET:
        llm = (
            AnthropicLLM(settings.anthropic_api_key, settings.llm_model)
            if settings.anthropic_api_key
            else None
        )
    app = FastAPI(
        title="Policy-to-Patient API (Phase 1 prototype)",
        description=(
            "Policy PDF → page-aware evidence → grounded Q&A with citations → synthetic cost "
            "estimate → deterministic coverage/OOP. Prototype only: outputs are estimates, not "
            "insurance decisions, and cost figures are synthetic."
        ),
        version="0.1.0",
    )
    app.state.policy_service = PolicyService(settings, PolicyStore(settings.data_dir), llm)  # type: ignore[arg-type]
    app.state.cost_estimator = cost_estimator or DemoTableEstimator(
        settings.cost_table_path, settings.cost_modifiers_path
    )
    app.include_router(router)
    return app


app = create_app()
