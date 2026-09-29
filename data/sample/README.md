# Sample data

| File | What it is | Provenance |
|---|---|---|
| `demo_treatment_costs.csv` | Procedure cost ranges for 8 treatments | **SYNTHETIC.** Hand-written round numbers for the prototype. Not real prices, not derived from any hospital, insurer or dataset. |
| `demo_cost_modifiers.csv` | City-tier multipliers and daily room rates | **SYNTHETIC.** Same as above. |

These files exist only so the end-to-end flow can be demonstrated. Every API
response that uses them carries `is_synthetic: true` and a disclaimer. They
will be replaced by a model trained on a real, provided dataset in Phase 2.

No policy PDF is committed. Use a real, publicly available policy wording for
demos (upload it through the API); tests build their own fictional PDF in code.
