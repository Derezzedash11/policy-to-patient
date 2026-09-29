"""Write the FICTIONAL evaluation policy as a PDF for demos.

    python -m evaluation.demo_pdf ../data/runtime/fictional_health_shield_policy.pdf

The document is fictional test data, not a real insurance policy.
"""

from __future__ import annotations

import sys
from pathlib import Path

from evaluation.dataset import load_dataset


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    out = Path(args[0]) if args else Path("fictional_health_shield_policy.pdf")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(load_dataset().policy_pdf())
    print(f"Wrote fictional demo policy to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
