"""Test fixtures: fictional policy text and the shared PDF builder."""

from __future__ import annotations

from evaluation.synthetic_pdf import make_text_pdf

__all__ = ["make_text_pdf", "FICTIONAL_POLICY_PAGES"]


# Fictional policy wording used only by tests. It is NOT a real insurance policy.
FICTIONAL_POLICY_PAGES: list[list[str]] = [
    [
        "FICTIONAL TEST POLICY - NOT A REAL INSURANCE PRODUCT",
        "1. DEFINITIONS",
        "Hospitalisation means admission to a hospital for at least 24 consecutive hours.",
        "Day care treatment means a procedure that needs less than 24 hours of admission.",
    ],
    [
        "2. COVERAGE AND LIMITS",
        "The sum insured is Rs 5,00,000 per policy year.",
        "Room rent is covered up to Rs 5,000 per day.",
        "Cataract surgery is subject to a sub-limit of Rs 40,000 per eye.",
    ],
    [],  # page without a text layer
    [
        "3. DEDUCTIBLE AND CO-PAYMENT",
        "A deductible of Rs 10,000 applies to each policy year.",
        "The insured person bears a co-payment of 10 percent of every admissible claim.",
        "4. WAITING PERIODS",
        "Maternity expenses are covered after a waiting period of 24 months.",
        "5. EXCLUSIONS",
        "Cosmetic surgery and dental treatment are excluded unless caused by an accident.",
    ],
]
