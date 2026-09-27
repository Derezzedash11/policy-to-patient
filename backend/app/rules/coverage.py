"""Deterministic coverage / out-of-pocket calculation.

Pure Python, no LLM. Rules are applied in a fixed, documented order:

1. Exclusion            - excluded category → nothing payable
2. Waiting period       - policy younger than the category's waiting period → nothing payable
3. Room-rent cap        - daily room charge above the limit is paid by the patient
4. Category sub-limit   - payable amount capped at the category sub-limit
5. Deductible           - first part of the payable amount is paid by the patient
6. Co-pay               - patient pays a percentage of what remains
7. Coverage limit       - insurer pays at most the remaining sum insured

The order is a stated modelling assumption; real policies may differ (e.g.
proportionate deductions on room rent are not modelled). Missing terms are
listed and never filled with guessed values.
"""

from __future__ import annotations

from decimal import ROUND_HALF_UP, Decimal

from app.models import CalculationStep, CoverageResult, PolicyTerms, TermCitation

DISCLAIMER = (
    "Estimate only, calculated deterministically from the policy terms you supplied. "
    "It is not an insurance decision or a guarantee of payment; the insurer's assessment prevails."
)
REQUIRED_TERMS = ("sum_insured", "deductible", "copay_pct")

_CENT = Decimal("0.01")


def _d(value: float | int) -> Decimal:
    return Decimal(str(value)).quantize(_CENT, ROUND_HALF_UP)


def _money(value: Decimal) -> float:
    return float(value.quantize(_CENT, ROUND_HALF_UP))


class _Ledger:
    def __init__(self, terms: PolicyTerms) -> None:
        self.steps: list[CalculationStep] = []
        self._terms = terms

    def add(self, rule: str, description: str, before: Decimal, after: Decimal, term: str | None) -> None:
        self.steps.append(
            CalculationStep(
                step=len(self.steps) + 1,
                rule=rule,
                description=description,
                amount_before=_money(before),
                amount_after=_money(after),
                patient_share_added=_money(before - after),
                citation=self._terms.citations.get(term) if term else None,
            )
        )


def calculate_coverage(
    total_cost: float,
    terms: PolicyTerms,
    category: str | None = None,
    room_rent_per_day: float | None = None,
    length_of_stay_days: int | None = None,
) -> CoverageResult:
    if total_cost < 0:
        raise ValueError("total_cost must be non-negative")
    total = _d(total_cost)
    payable = total
    ledger = _Ledger(terms)
    missing = [name for name in REQUIRED_TERMS if getattr(terms, name) is None]
    assumptions: list[str] = []

    def result(status: str) -> CoverageResult:
        return CoverageResult(
            status=status,  # type: ignore[arg-type]
            total_cost=_money(total),
            covered_amount=_money(payable),
            out_of_pocket=_money(total - payable),
            steps=ledger.steps,
            missing_terms=missing,
            assumptions=assumptions,
            disclaimer=DISCLAIMER,
        )

    # 1. Exclusions
    if category is None:
        if terms.excluded_categories or terms.sub_limits or terms.waiting_period_months:
            assumptions.append(
                "Treatment category unknown: exclusions, waiting periods and sub-limits were not checked."
            )
    elif category in terms.excluded_categories:
        ledger.add("exclusion", f"Category '{category}' is excluded by the policy.", payable, Decimal(0),
                   "excluded_categories")
        payable = Decimal(0)
        return result("not_covered")

    # 2. Waiting period
    waiting = terms.waiting_period_months.get(category) if category else None
    if waiting is not None:
        if terms.policy_age_months is None:
            missing.append("policy_age_months")
            assumptions.append(
                f"Waiting period of {waiting} months for '{category}' not evaluated: policy age unknown."
            )
        elif terms.policy_age_months < waiting:
            ledger.add(
                "waiting_period",
                f"Policy age {terms.policy_age_months} months is within the {waiting}-month waiting "
                f"period for '{category}'.",
                payable, Decimal(0), "waiting_period_months",
            )
            payable = Decimal(0)
            return result("not_covered")

    # 3. Room-rent cap
    if terms.room_rent_limit_per_day is not None:
        if room_rent_per_day is None or length_of_stay_days is None:
            assumptions.append("Room-rent limit not applied: room charge per day or length of stay unknown.")
        else:
            excess_per_day = max(Decimal(0), _d(room_rent_per_day) - _d(terms.room_rent_limit_per_day))
            excess = min(payable, excess_per_day * length_of_stay_days)
            before, payable = payable, payable - excess
            ledger.add(
                "room_rent_cap",
                f"Room charge {_money(_d(room_rent_per_day))}/day vs limit "
                f"{_money(_d(terms.room_rent_limit_per_day))}/day for {length_of_stay_days} day(s); "
                f"excess {_money(excess)} is not payable.",
                before, payable, "room_rent_limit_per_day",
            )
            assumptions.append("Only the room-rent excess is deducted; proportionate deductions are not modelled.")

    # 4. Category sub-limit
    if category is not None and category in terms.sub_limits:
        limit = _d(terms.sub_limits[category])
        before, payable = payable, min(payable, limit)
        ledger.add("sub_limit", f"Sub-limit for '{category}' is {_money(limit)}.", before, payable, "sub_limits")

    # 5. Deductible
    if terms.deductible is not None:
        remaining = max(Decimal(0), _d(terms.deductible) - _d(terms.deductible_already_met))
        applied = min(remaining, payable)
        before, payable = payable, payable - applied
        ledger.add(
            "deductible",
            f"Deductible {_money(_d(terms.deductible))} (already met: "
            f"{_money(_d(terms.deductible_already_met))}); patient pays {_money(applied)}.",
            before, payable, "deductible",
        )
    else:
        assumptions.append("Deductible not provided: none applied, so out-of-pocket may be understated.")

    # 6. Co-pay
    if terms.copay_pct is not None:
        share = (payable * Decimal(str(terms.copay_pct)) / 100).quantize(_CENT, ROUND_HALF_UP)
        before, payable = payable, payable - share
        ledger.add("copay", f"Co-pay {terms.copay_pct}% of {_money(before)} = {_money(share)}.",
                   before, payable, "copay_pct")
    else:
        assumptions.append("Co-pay not provided: none applied, so out-of-pocket may be understated.")

    # 7. Coverage limit (sum insured)
    if terms.sum_insured is not None:
        available = max(Decimal(0), _d(terms.sum_insured) - _d(terms.sum_insured_already_used))
        before, payable = payable, min(payable, available)
        ledger.add(
            "coverage_limit",
            f"Remaining sum insured is {_money(available)} "
            f"(sum insured {_money(_d(terms.sum_insured))} minus {_money(_d(terms.sum_insured_already_used))} used).",
            before, payable, "sum_insured",
        )
    else:
        assumptions.append("Sum insured not provided: insurer payment was not capped.")

    return result("incomplete_terms" if missing else "complete")
