# Field-confidence calibration

Written by `jobx calibrate` (run by the eval-suite workflow after the fine-tune's Q4_K_M predictions with
token log-probs exist): `<approach>.json` (fit on val, reliability and risk-coverage on held-out splits)
and `calibrator.json` (the weights and threshold the API uses). Until then the API falls back to
evidence-only confidence.
