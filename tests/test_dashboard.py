import json
import shutil

from conftest import ROOT
from jobx.eval import dashboard as dash


def _root(tmp_path):
    for d in ("data", "results/predictions", "results/robustness", "results/audit"):
        (tmp_path / d).parent.mkdir(parents=True, exist_ok=True)
        if (ROOT / d).is_dir():
            shutil.copytree(ROOT / d, tmp_path / d)
    (tmp_path / "results/results.json").write_text((ROOT / "results/results.json").read_text())
    return tmp_path


def test_sections_render_from_committed_files_and_hide_when_missing(tmp_path):
    root = _root(tmp_path)
    data = json.loads((root / "results/results.json").read_text())
    assert "Robustness" in dash.robustness_table(root) and "Injection success" in dash.robustness_table(root)
    assert "Leakage audit passed" in dash.data_card(root)
    assert dash.calibration_card(root) == "" and dash.quant_card(root) == ""
    assert dash.field_heat(data, "test_synth").count("<tr") >= 3
    cmp = dash.comparison_data(root)
    assert (
        cmp and len(cmp["examples"]) == 6 and set(cmp["examples"][0]["ours"]["correct"]) >= {"role", "salary"}
    )


def test_calibration_and_quant_cards_use_the_numbers_in_the_files(tmp_path):
    root = _root(tmp_path)
    rel = {"ece": 0.05, "auroc": 0.9, "bins": [{"bin": [0.9, 1.0], "n": 10, "conf": 0.95, "acc": 0.9}]}
    cal = {
        "threshold": 0.8,
        "target_selective_accuracy": 0.95,
        "n_fields_val": 100,
        "splits": {
            "test_real": {
                "n_examples": 5,
                "n_fields": 55,
                "raw_token_probability": rel,
                "calibrated": rel,
                "at_threshold": {"coverage": 0.7, "selective_accuracy": 0.96, "accuracy_all_fields": 0.9},
            }
        },
    }
    (root / "results/calibration").mkdir(parents=True)
    (root / "results/calibration" / f"{dash.OURS}.json").write_text(json.dumps(cal))
    html = dash.calibration_card(root)
    assert "0.050 &rarr; 0.050" in html and "70.0% of fields" in html and "<svg" in html
    sweep = {
        "hardware": "CPU (x)",
        "llama_cpp": "b6000",
        "subsets": {"test_synth": 150, "test_real": 130},
        "date": "d",
        "commit": "c",
        "variants": [
            {
                "quant": "Q4_K_M",
                "size_gb": 0.4,
                "gen_tokens_per_s": 50.0,
                "prompt_tokens_per_s": 300.0,
                "warm_latency_s": {"p50": 3.1},
                "startup_s": 0.9,
                "peak_ram_gb": 0.5,
                "accuracy": {
                    "test_synth": {"micro_f1": 0.99, "json_validity": 1.0},
                    "test_real": {"micro_f1": 0.7},
                },
            }
        ],
        "thread_sweep": [
            {"quant": "Q4_K_M", "threads": 1, "gen_tokens_per_s": 15.0, "warm_latency_s": {"p50": 9.0}}
        ],
    }
    (root / "results/quant").mkdir(parents=True)
    (root / "results/quant/sweep.json").write_text(json.dumps(sweep))
    q = dash.quant_card(root)
    assert "70.0" in q and "99.0" in q and "1 thread: 15.0 tok/s" in q and "browser" in q
