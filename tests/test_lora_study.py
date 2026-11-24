import json

from jobx.cpu.study import load_study, matrix, resolve, summary


def test_every_config_shares_the_same_budget():
    s = load_study()
    ids = [c["id"] for c in s["configs"]]
    assert len(ids) == len(set(ids)) and [m["id"] for m in matrix(s)] == ids
    budgets = {
        (
            resolve(s, i)["n_train"],
            resolve(s, i)["max_steps"],
            resolve(s, i)["batch_size"] * resolve(s, i)["grad_accum"],
        )
        for i in ids
    }
    assert budgets == {(640, 40, 16)}
    assert resolve(s, "r8_all")["r"] == 8 and resolve(s, "r16_attn")["target_modules"] == [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
    ]
    assert "Reduced-budget" in s["note"]


def test_summary_selects_on_validation_and_reports_ties(tmp_path):
    def run(i, f1, ci, params):
        return {
            "id": i,
            "config": {},
            "params_trainable": params,
            "trainable_share": 0.01,
            "adapter_mb": 1.0,
            "train_seconds": 1,
            "seconds_per_step": 1,
            "peak_rss_gb": 1,
            "final_train_loss": 0.1,
            "val_loss": 0.1,
            "cpu": "x",
            "bf16": False,
            "val": {"micro_f1": f1, "ci95_micro_f1": ci},
        }

    for r in (
        run("a", 0.90, [0.88, 0.92], 10),
        run("b", 0.89, [0.87, 0.91], 5),
        run("c", 0.70, [0.65, 0.75], 5),
    ):
        (tmp_path / f"{r['id']}.json").write_text(json.dumps(r))
    res = summary(tmp_path, {"note": "n"})
    assert (
        res["best"] == "a"
        and res["within_best_ci"] == ["b"]
        and [r["id"] for r in res["runs"]] == ["a", "b", "c"]
    )
