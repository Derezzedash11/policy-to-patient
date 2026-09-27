"""Cost estimator interface. Phase 2 can plug in a trained ML model."""

from __future__ import annotations

from typing import Protocol

from app.models import CostEstimate, Treatment, TreatmentInput


class UnknownTreatmentError(KeyError):
    pass


class CostEstimator(Protocol):
    def list_treatments(self) -> list[Treatment]: ...

    def estimate(self, request: TreatmentInput) -> CostEstimate:
        """Raise UnknownTreatmentError for treatments the estimator does not cover."""
