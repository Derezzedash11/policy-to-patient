import pytest

from app.config import Settings
from app.cost.base import UnknownTreatmentError
from app.cost.demo_table import DemoTableEstimator
from app.models import TreatmentInput


@pytest.fixture
def estimator():
    s = Settings()
    return DemoTableEstimator(s.cost_table_path, s.cost_modifiers_path)


def test_lists_treatments(estimator):
    codes = {t.treatment_code for t in estimator.list_treatments()}
    assert {"APPENDECTOMY", "CATARACT", "KNEE_REPLACEMENT"} <= codes


def test_estimate_is_labelled_synthetic(estimator):
    est = estimator.estimate(TreatmentInput(treatment_code="APPENDECTOMY"))
    assert est.is_synthetic is True
    assert "SYNTHETIC" in est.disclaimer
    assert "synthetic" in est.source


def test_estimate_arithmetic(estimator):
    # From the synthetic table: procedure 60000-110000, semi_private room 4000-7000/day, tier1 x1.0
    est = estimator.estimate(
        TreatmentInput(treatment_code="APPENDECTOMY", city_tier="tier1", room_type="semi_private",
                       length_of_stay_days=2)
    )
    assert est.low == 60000 + 4000 * 2
    assert est.high == 110000 + 7000 * 2
    assert est.estimate == (est.low + est.high) / 2
    assert est.room_rent_per_day == 5500
    assert sum(c.low for c in est.components) == est.low


def test_default_length_of_stay_and_tier_multiplier(estimator):
    t1 = estimator.estimate(TreatmentInput(treatment_code="HERNIA_REPAIR", city_tier="tier1"))
    t3 = estimator.estimate(TreatmentInput(treatment_code="HERNIA_REPAIR", city_tier="tier3"))
    assert t1.length_of_stay_days == 2
    assert t3.low < t1.low and t3.high < t1.high
    assert t1.low <= t1.estimate <= t1.high


def test_zero_day_stay_has_no_room_cost(estimator):
    est = estimator.estimate(TreatmentInput(treatment_code="CATARACT"))
    assert est.length_of_stay_days == 0
    assert est.components[1].low == est.components[1].high == 0


def test_unknown_treatment(estimator):
    with pytest.raises(UnknownTreatmentError):
        estimator.estimate(TreatmentInput(treatment_code="NOPE"))
