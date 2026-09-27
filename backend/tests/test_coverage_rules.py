import pytest

from app.models import PolicyTerms, TermCitation
from app.rules.coverage import calculate_coverage


def full_terms(**overrides) -> PolicyTerms:
    base = dict(sum_insured=500000, deductible=10000, copay_pct=10)
    base.update(overrides)
    return PolicyTerms(**base)


def rules(result):
    return [s.rule for s in result.steps]


def test_basic_deductible_then_copay():
    r = calculate_coverage(100000, full_terms())
    # 100000 - 10000 deductible = 90000; 10% co-pay = 9000; insurer 81000
    assert r.status == "complete"
    assert r.covered_amount == 81000
    assert r.out_of_pocket == 19000
    assert r.covered_amount + r.out_of_pocket == r.total_cost
    assert rules(r) == ["deductible", "copay", "coverage_limit"]
    assert r.steps[0].patient_share_added == 10000
    assert r.steps[1].patient_share_added == 9000


def test_cost_below_deductible_pays_nothing():
    r = calculate_coverage(6000, full_terms())
    assert r.covered_amount == 0
    assert r.out_of_pocket == 6000


def test_deductible_partly_met():
    r = calculate_coverage(50000, full_terms(deductible_already_met=7000, copay_pct=0))
    assert r.covered_amount == 47000


def test_deductible_fully_met():
    r = calculate_coverage(50000, full_terms(deductible_already_met=15000, copay_pct=0))
    assert r.covered_amount == 50000


def test_zero_and_full_copay():
    assert calculate_coverage(50000, full_terms(deductible=0, copay_pct=0)).covered_amount == 50000
    assert calculate_coverage(50000, full_terms(deductible=0, copay_pct=100)).covered_amount == 0


def test_coverage_limit_caps_insurer_payment():
    r = calculate_coverage(1000000, full_terms(sum_insured=300000))
    # 1000000 - 10000 = 990000; co-pay 99000 → 891000; capped at 300000
    assert r.covered_amount == 300000
    assert r.out_of_pocket == 700000
    assert r.steps[-1].rule == "coverage_limit"


def test_coverage_limit_accounts_for_prior_claims():
    r = calculate_coverage(100000, full_terms(sum_insured=100000, sum_insured_already_used=80000))
    assert r.covered_amount == 20000


def test_exhausted_sum_insured():
    r = calculate_coverage(100000, full_terms(sum_insured=50000, sum_insured_already_used=60000))
    assert r.covered_amount == 0


def test_sub_limit_applies_before_deductible():
    r = calculate_coverage(60000, full_terms(sub_limits={"cataract": 40000}), category="cataract")
    # min(60000, 40000) = 40000; -10000 = 30000; -10% = 27000
    assert rules(r)[0] == "sub_limit"
    assert r.covered_amount == 27000


def test_sub_limit_ignored_for_other_category():
    r = calculate_coverage(60000, full_terms(sub_limits={"cataract": 40000}), category="surgery")
    assert "sub_limit" not in rules(r)


def test_room_rent_excess_paid_by_patient():
    terms = full_terms(room_rent_limit_per_day=5000, deductible=0, copay_pct=0)
    r = calculate_coverage(100000, terms, category="surgery", room_rent_per_day=7000, length_of_stay_days=3)
    assert r.steps[0].rule == "room_rent_cap"
    assert r.steps[0].patient_share_added == 6000
    assert r.covered_amount == 94000


def test_room_rent_within_limit_no_deduction():
    terms = full_terms(room_rent_limit_per_day=5000)
    r = calculate_coverage(100000, terms, room_rent_per_day=4000, length_of_stay_days=3)
    assert r.steps[0].patient_share_added == 0


def test_room_rent_unknown_is_an_assumption_not_a_guess():
    r = calculate_coverage(100000, full_terms(room_rent_limit_per_day=5000))
    assert "room_rent_cap" not in rules(r)
    assert any("Room-rent limit not applied" in a for a in r.assumptions)


def test_excluded_category_not_covered():
    r = calculate_coverage(80000, full_terms(excluded_categories=["cosmetic"]), category="cosmetic")
    assert r.status == "not_covered"
    assert r.covered_amount == 0
    assert r.out_of_pocket == 80000


def test_waiting_period_not_served():
    terms = full_terms(waiting_period_months={"maternity": 24}, policy_age_months=12)
    r = calculate_coverage(80000, terms, category="maternity")
    assert r.status == "not_covered"
    assert r.covered_amount == 0


def test_waiting_period_served():
    terms = full_terms(waiting_period_months={"maternity": 24}, policy_age_months=30)
    r = calculate_coverage(80000, terms, category="maternity")
    assert r.status == "complete"
    assert r.covered_amount == 63000


def test_waiting_period_with_unknown_policy_age_is_flagged():
    r = calculate_coverage(80000, full_terms(waiting_period_months={"maternity": 24}), category="maternity")
    assert r.status == "incomplete_terms"
    assert "policy_age_months" in r.missing_terms


def test_missing_terms_are_listed_not_guessed():
    r = calculate_coverage(100000, PolicyTerms())
    assert r.status == "incomplete_terms"
    assert set(r.missing_terms) == {"sum_insured", "deductible", "copay_pct"}
    assert r.steps == []
    assert len(r.assumptions) == 3


def test_rounding_to_cents():
    r = calculate_coverage(33333.33, full_terms(deductible=0, copay_pct=12.5))
    assert r.covered_amount == 29166.66  # 33333.33 * 0.875 = 29166.664 → co-pay 4166.67
    assert round(r.covered_amount + r.out_of_pocket, 2) == 33333.33


def test_zero_cost():
    r = calculate_coverage(0, full_terms())
    assert r.covered_amount == 0 and r.out_of_pocket == 0


def test_negative_cost_rejected():
    with pytest.raises(ValueError):
        calculate_coverage(-1, full_terms())


def test_citations_are_attached_to_steps():
    cite = TermCitation(chunk_id="doc:p4:c5", page=4, section="3. DEDUCTIBLE AND CO-PAYMENT")
    r = calculate_coverage(100000, full_terms(citations={"deductible": cite}))
    deductible_step = next(s for s in r.steps if s.rule == "deductible")
    assert deductible_step.citation == cite
    assert next(s for s in r.steps if s.rule == "copay").citation is None


def test_deterministic():
    terms = full_terms(sub_limits={"surgery": 70000}, room_rent_limit_per_day=3000)
    args = (90000, terms, "surgery", 4500, 2)
    assert calculate_coverage(*args) == calculate_coverage(*args)
