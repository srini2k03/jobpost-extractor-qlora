import json

import pytest
from rich.console import Console

from jobx.cards import PENDING, metrics_table, render_card
from jobx.data.fetch_real import looks_indian, select_rows
from jobx.hub import PLACEHOLDER, hf_username
from jobx.train.config import load_train_config, parse_overrides, sweep_runs
from jobx.verify import verify


def test_train_config_inherit_and_overrides():
    q = load_train_config("configs/train/qwen2.5-1.5b-qlora.yaml")
    assert q["lora"]["r"] == 16 and q["lora"]["alpha"] == 32
    t = q["training"]
    assert t["per_device_train_batch_size"] * t["gradient_accumulation_steps"] == 16
    assert t["fp16"] and not t["bf16"] and q["quantization"]["bnb_4bit_quant_type"] == "nf4"
    ll = load_train_config("configs/train/llama3.2-1b-qlora.yaml")
    assert ll["base_model"].startswith("meta-llama/") and ll["lora"]["r"] == 16
    o = load_train_config(
        "configs/train/qwen2.5-1.5b-qlora.yaml", parse_overrides(["lora.r=8", "training.learning_rate=1e-4"])
    )
    assert o["lora"]["r"] == 8 and float(o["training"]["learning_rate"]) == 1e-4
    runs = dict(sweep_runs("configs/train/sweep.yaml"))
    assert set(runs) == {"r8-lr2e-4", "r16-lr2e-4", "r32-lr2e-4", "r16-lr1e-4"}
    assert (
        runs["r32-lr2e-4"]["lora"]["alpha"] == 64 and runs["r32-lr2e-4"]["training"]["num_train_epochs"] == 1
    )


def test_hub_username_resolution(monkeypatch):
    with pytest.raises(RuntimeError):
        hf_username()
    monkeypatch.setenv("HF_USERNAME", "someone")
    assert hf_username() == "someone"
    from jobx.hub import repo_id

    assert repo_id("gguf") == "someone/jobx-qwen2.5-1.5b-gguf"
    assert PLACEHOLDER == "YOUR_HF_USERNAME"


def test_model_cards_render():
    card = render_card(
        "adapter",
        {"base_model": "Qwen/Qwen2.5-1.5B-Instruct", "metrics_table": metrics_table(["qwen2.5-1.5b_qlora"])},
    )
    assert "Qwen/Qwen2.5-1.5B-Instruct" in card and PENDING in card
    assert "| regex_baseline | test_synth |" in metrics_table(["regex_baseline"])
    for kind in ("adapter", "merged", "gguf", "awq"):
        assert render_card(kind, {}).startswith("---\n")


def test_fetch_real_selection():
    rows = [
        {"description": "x" * 50, "location": "Bengaluru"},  # too short
        {"description": "Backend role in Austin, TX. " * 20, "location": "Austin"},  # not Indian
        {"description": "Backend engineer, CTC 12-18 LPA, Python. " * 10, "location": ""},
        {"description": "Backend engineer, CTC 12-18 LPA, Python. " * 10, "location": ""},  # duplicate
        {"description": "Sales role, great team. " * 15, "location": "Pune, Maharashtra, India"},
    ]
    cand = {"text_columns": ["description"], "location_columns": ["location"]}
    cfg = {"min_chars": 200, "max_chars": 3000, "n_target": 10, "seed": 1}
    picked = select_rows(rows, cand, cfg)
    assert [p["row"] for p in picked] == [2, 4]
    assert looks_indian("Salary 5 LPA") and not looks_indian("Salary $50k", "Austin")


def test_verify_cli_flow(tmp_path):
    drafts = tmp_path / "d.jsonl"
    out = tmp_path / "test_real.jsonl"
    base = {
        "company": None,
        "role": "Telecaller",
        "seniority": None,
        "skills": [],
        "min_experience_years": None,
        "max_experience_years": None,
        "salary": None,
        "locations": [],
        "work_mode": None,
        "employment_type": None,
        "education": None,
    }
    rows = [
        {
            "id": f"real-{i}",
            "text": f"posting {i}",
            "label": base,
            "meta": {"dataset": "x/y", "licence": "mit"},
        }
        for i in range(3)
    ]
    drafts.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    answers = iter(["e", "locations", '["Chennai"]', "e", "work_mode", '"wfh"', "a", "a", "s"])
    meta = verify(drafts, out, input_fn=lambda _: next(answers), console=Console(quiet=True))
    got = [json.loads(x) for x in out.read_text().splitlines()]
    assert [g["id"] for g in got] == ["real-0", "real-1"]
    assert got[0]["label"]["locations"] == ["Chennai"] and got[0]["meta"]["edited_fields"] == ["locations"]
    assert got[0]["label"]["work_mode"] is None  # invalid edit rejected
    assert meta["n"] == 2 and meta["this_session"]["skipped"] == 1
    # resume: nothing left to do
    meta2 = verify(drafts, out, input_fn=lambda _: "q", console=Console(quiet=True))
    assert meta2["n"] == 2
