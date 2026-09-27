"""Cost estimator backed by a small SYNTHETIC lookup table (not real prices)."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from app.cost.base import UnknownTreatmentError
from app.models import CostComponent, CostEstimate, Treatment, TreatmentInput

DISCLAIMER = (
    "SYNTHETIC DEMO ESTIMATE: computed from hand-written placeholder numbers, not real "
    "hospital prices or a trained model. Do not use it for any financial or medical decision."
)


@dataclass(frozen=True)
class _Row:
    treatment: Treatment
    procedure_low: float
    procedure_high: float


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        lines = [line for line in fh if line.strip() and not line.lstrip().startswith("#")]
    return list(csv.DictReader(lines))


def _r(value: float) -> float:
    return round(value, 2)


class DemoTableEstimator:
    currency = "INR"

    def __init__(self, table_path: Path, modifiers_path: Path) -> None:
        self._source = f"{table_path.name} + {modifiers_path.name} (synthetic)"
        self._rows: dict[str, _Row] = {}
        for rec in _read_csv(table_path):
            t = Treatment(
                treatment_code=rec["treatment_code"],
                treatment_name=rec["treatment_name"],
                category=rec["category"],
                typical_length_of_stay_days=int(rec["typical_length_of_stay_days"]),
            )
            self._rows[t.treatment_code] = _Row(
                t, float(rec["procedure_cost_low"]), float(rec["procedure_cost_high"])
            )
        self._tier: dict[str, tuple[float, float]] = {}
        self._room: dict[str, tuple[float, float]] = {}
        for rec in _read_csv(modifiers_path):
            target = self._tier if rec["kind"] == "city_tier_multiplier" else self._room
            target[rec["key"]] = (float(rec["low"]), float(rec["high"]))

    def list_treatments(self) -> list[Treatment]:
        return [row.treatment for row in self._rows.values()]

    def estimate(self, request: TreatmentInput) -> CostEstimate:
        row = self._rows.get(request.treatment_code)
        if row is None:
            raise UnknownTreatmentError(request.treatment_code)
        t = row.treatment
        los = (
            request.length_of_stay_days
            if request.length_of_stay_days is not None
            else t.typical_length_of_stay_days
        )
        m_low, m_high = self._tier[request.city_tier]
        room_low, room_high = self._room[request.room_type]

        procedure = CostComponent(
            name="procedure", low=_r(row.procedure_low * m_low), high=_r(row.procedure_high * m_high)
        )
        room = CostComponent(
            name=f"room ({request.room_type}, {los} day(s))",
            low=_r(room_low * m_low * los),
            high=_r(room_high * m_high * los),
        )
        low = _r(procedure.low + room.low)
        high = _r(procedure.high + room.high)
        return CostEstimate(
            treatment_code=t.treatment_code,
            treatment_name=t.treatment_name,
            category=t.category,
            city_tier=request.city_tier,
            room_type=request.room_type,
            length_of_stay_days=los,
            currency=self.currency,
            low=low,
            high=high,
            estimate=_r((low + high) / 2),
            room_rent_per_day=_r((room_low * m_low + room_high * m_high) / 2),
            components=[procedure, room],
            method="lookup: (procedure range + daily room rate x stay) x city-tier multiplier",
            is_synthetic=True,
            source=self._source,
            disclaimer=DISCLAIMER,
        )
