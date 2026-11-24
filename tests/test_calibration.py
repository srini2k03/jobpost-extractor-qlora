import json
import random

from conftest import ROOT
from jobx.data.build import read_jsonl
from jobx.eval.calibration import calibrate, field_records
from jobx.uncertainty import value_spans


def _write_preds(root, split, rng, *, n=60):
    rows = read_jsonl(ROOT / f"data/processed/{split}.jsonl")[:n]
    out = []
    for r in rows:
        lab = dict(r["label"])
        wrong = rng.random() < 0.3
        if wrong:
            lab["role"] = "Totally Wrong Role"
        text = json.dumps(lab)
        lps = {}
        for k in value_spans(text):
            conf = -2.5 if (wrong and k == "role") else -0.05
            lps[k] = {"mean": conf + rng.uniform(-0.05, 0.05), "min": conf - 0.1, "sum": conf, "n": 2}
        out.append({"id": r["id"], "output": text, "field_logprobs": lps})
    d = root / "results/predictions"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"toy__{split}.jsonl").write_text("".join(json.dumps(o) + "\n" for o in out))


def test_calibrate_fits_on_val_and_reports_held_out(tmp_path):
    rng = random.Random(0)
    (tmp_path / "data").symlink_to(ROOT / "data")
    _write_preds(tmp_path, "val", rng)
    _write_preds(tmp_path, "test_synth", rng)
    recs = field_records("toy", "val", tmp_path)
    assert len(recs) == 60 * 11 and {r["field"] for r in recs} >= {"role", "salary"}
    rep = calibrate("toy", tmp_path, target=0.99)
    assert rep["fitted_on"] == "val" and set(rep["splits"]) == {"val", "test_synth"}
    t = rep["splits"]["test_synth"]
    assert t["calibrated"]["auroc"] > 0.9
    assert t["at_threshold"]["selective_accuracy"] >= t["at_threshold"]["accuracy_all_fields"]
    saved = json.loads((tmp_path / "results/calibration/calibrator.json").read_text())
    assert saved["approach"] == "toy" and len(saved["weights"]) == 7
