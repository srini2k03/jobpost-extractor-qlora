"""CPU route (no network, no torch): planner, matrices, prompts, llama-server client, site/demo, workflows."""

import json
import os
import re
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from jobx import prompts
from jobx.cpu import matrix
from jobx.cpu.card import render_model_card
from jobx.cpu.plan import choose_max_length, plan_training, stratified_subset
from jobx.cpu.train import latest_checkpoint, load_route_config, status
from jobx.data.build import read_jsonl
from jobx.eval.results_table import build_tables
from jobx.eval.site import WLLAMA_VERSION, build_site

WF = Path(".github/workflows")


def test_route_config():
    cfg = load_route_config()
    assert cfg["base_model"] == "Qwen/Qwen2.5-0.5B-Instruct"
    assert cfg["lora"]["r"] == 16 and cfg["lora"]["alpha"] == 32 and len(cfg["lora"]["target_modules"]) == 7
    t = cfg["training"]
    assert t["per_device_train_batch_size"] * t["gradient_accumulation_steps"] == 16
    assert cfg["system_prompt"] == "compact" and t["max_length_target"] == 512
    assert cfg["budget"]["job_minutes"] < 340  # leaves room inside the 6 h job limit


def test_choose_max_length():
    assert choose_max_length([300] * 200 + [700]) == 512
    assert choose_max_length(list(range(400, 600))) == 640
    assert choose_max_length([900] * 10, cap=768) == 768


@pytest.mark.parametrize(
    "sps, expect_epochs, expect_n",
    [(60.0, 2, 2000), (250.0, 1, 2000), (600.0, 1, None)],
)
def test_plan_training(sps, expect_epochs, expect_n):
    p = plan_training(
        seconds_per_step=sps,
        effective_batch=16,
        n_available=2000,
        max_length=512,
        job_minutes=300,
        max_jobs=3,
        overhead=0.85,
        max_epochs=2,
    )
    assert p.epochs == expect_epochs
    if expect_n:
        assert p.n_train == expect_n
    else:
        assert p.n_train < 2000 and p.n_train % 16 == 0 and "subset" in p.reason
    assert p.est_hours * 3600 <= 3 * 300 * 60 * 0.85 + 1
    assert 1 <= p.jobs_needed <= 3
    assert p.max_steps == -(-p.n_train * p.epochs // 16)


def test_stratified_subset_is_proportional_and_deterministic():
    train = read_jsonl("data/processed/train.jsonl")
    a = stratified_subset(train, 500)
    assert len(a) == 500 and a == stratified_subset(train, 500)

    def fam(xs):
        return {
            f: sum(e["meta"]["family"] == f for e in xs) / len(xs)
            for f in {e["meta"]["family"] for e in train}
        }

    full, sub = fam(train), fam(a)
    assert all(abs(full[f] - sub.get(f, 0)) < 0.02 for f in full)
    assert stratified_subset(train, 10**6) == list(train)


def test_compact_prompt_used_for_finetune_only():
    msgs = prompts.messages("x", system="compact")
    assert msgs[0]["content"] == prompts.COMPACT_SYSTEM_PROMPT and len(prompts.COMPACT_SYSTEM_PROMPT) < 400
    assert prompts.messages("x")[0]["content"] == prompts.SYSTEM_PROMPT
    rec = prompts.sft_record(read_jsonl("data/processed/train.jsonl")[0], system="compact")
    assert rec["prompt"][0]["content"] == prompts.COMPACT_SYSTEM_PROMPT
    for k in ("company", "salary", "education"):
        assert k in prompts.COMPACT_SYSTEM_PROMPT


def test_matrices():
    cfg = load_route_config()
    b = matrix.baselines_matrix(cfg)
    ids = {e["approach"] for e in b}
    assert {
        "qwen2.5-0.5b-cpu_zero_shot",
        "qwen2.5-0.5b-cpu_few_shot_guided",
        "qwen2.5-1.5b-cpu_few_shot",
    } <= ids
    for e in b:
        assert e["shots"] == (3 if "few_shot" in e["approach"] else 0)
        assert e["max_examples"] == (200 if e["guided"] else 0)
    ev = matrix.eval_matrix(cfg)
    assert [e["quant"] for e in ev] == ["F16", "Q8_0", "Q4_K_M"]
    assert ev[2]["approach"] == "qwen2.5-0.5b-lora-cpu_q4_k_m"
    approaches = {a["id"] for a in yaml.safe_load(Path("configs/approaches.yaml").read_text())["approaches"]}
    assert ids | {e["approach"] for e in ev} <= approaches, "every CPU-route row must be in approaches.yaml"
    assert matrix.available_splits() == ["test_synth"]


def test_train_state_helpers(tmp_path):
    for n in (20, 100, 40):
        (tmp_path / f"checkpoint-{n}").mkdir()
    assert latest_checkpoint(tmp_path).name == "checkpoint-100"
    assert latest_checkpoint(tmp_path / "nope") is None
    cfg = {**load_route_config(), "output_dir": str(tmp_path / "state")}
    assert status(cfg, root=Path("/")) == {"done": False, "global_step": 0, "final_adapter": False}


def test_llama_server_client_request_shapes(monkeypatch, tmp_path):
    """benchmark.run talks to llama-server; fake the process and HTTP client."""
    import httpx

    from jobx.quantize import benchmark as bm

    sent = []
    gold = {e["id"]: e for e in read_jsonl("data/processed/test_synth.jsonl")}

    class FakeClient:
        def __init__(self, *a, **k):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def post(self, url, json):
            sent.append(json)
            text = json["messages"][-1]["content"]
            ex = next(
                e
                for e in gold.values()
                if f"<posting>\n{e['text'].strip()}\n</posting>" == text.split("\n\n", 1)[1]
            )
            body = {
                "choices": [{"message": {"content": prompts.target_json(ex["label"])}}],
                "usage": {"completion_tokens": 120},
                "timings": {"predicted_per_second": 33.0},
            }
            return SimpleNamespace(raise_for_status=lambda: None, json=lambda: body)

    monkeypatch.setattr(httpx, "Client", FakeClient)
    monkeypatch.setattr(bm, "wait_ready", lambda url, proc: None)
    monkeypatch.setattr(bm, "git_commit", lambda root=".": "test")
    monkeypatch.setattr(
        bm.subprocess,
        "Popen",
        lambda *a, **k: SimpleNamespace(
            pid=os.getpid(), terminate=lambda: None, wait=lambda timeout: 0, poll=lambda: None
        ),
    )
    model = tmp_path / "m.gguf"
    model.write_bytes(b"0" * 1000)
    args = [
        "--server-bin",
        "x",
        "--model",
        str(model),
        "--approach",
        "qwen2.5-0.5b-lora-cpu_q4_k_m",
        "--split",
        "test_synth",
        "--max-examples",
        "5",
        "--system",
        "compact",
        "--out-pred",
        str(tmp_path / "pred"),
        "--out-bench",
        str(tmp_path / "bench.json"),
        "--variant",
        "LoRA 0.5B GGUF Q4_K_M (llama.cpp)",
        "--guided",
    ]
    ns = _parse(bm, args)
    row = bm.run(ns)
    assert len(sent) == 5
    assert (
        sent[0]["messages"][0]["content"] == prompts.COMPACT_SYSTEM_PROMPT and len(sent[0]["messages"]) == 2
    )
    assert sent[0]["response_format"]["type"] == "json_schema" and sent[0]["cache_prompt"] is True
    assert row["micro_f1"] == pytest.approx(1.0) and row["tokens_per_s"] == 33.0
    meta = json.loads((tmp_path / "pred" / "qwen2.5-0.5b-lora-cpu_q4_k_m__test_synth.meta.json").read_text())
    assert meta["system_prompt"] == "compact" and meta["decoding"].startswith("guided")
    # few-shot, predictions only (no --variant -> no benchmark row)
    sent.clear()
    ns = _parse(
        bm,
        [
            "--server-bin",
            "x",
            "--model",
            str(model),
            "--approach",
            "qwen2.5-0.5b-cpu_few_shot",
            "--split",
            "test_synth",
            "--max-examples",
            "2",
            "--shots",
            "3",
            "--out-pred",
            str(tmp_path / "pred"),
        ],
    )
    out = bm.run(ns)
    assert "predictions" in out and len(sent[0]["messages"]) == 8 and "response_format" not in sent[0]


def _parse(bm, argv):
    import argparse

    captured = {}

    def fake_run(a):
        captured["a"] = a
        return {}

    orig = bm.run
    bm.run = fake_run
    try:
        bm.main(argv)
    finally:
        bm.run = orig
    assert isinstance(captured["a"], argparse.Namespace)
    return captured["a"]


def test_site_with_demo(tmp_path):
    parts = tmp_path / "parts"
    parts.mkdir()
    for i in (0, 1):
        (parts / f"jobx-q4km.gguf.part00{i}").write_bytes(b"x" * 1000)
    out = tmp_path / "site"
    build_site(".", out, model_dir=parts)
    html = (out / "index.html").read_text()
    assert 'src="demo/demo.js"' in html and "Load model" in html and "<h2>Results</h2>" in html
    man = json.loads((out / "model" / "manifest.json").read_text())
    assert man["parts"] == ["model/jobx-q4km.gguf.part000", "model/jobx-q4km.gguf.part001"]
    assert man["file"] == "jobx-q4km.gguf"
    js = (out / "demo" / "demo.js").read_text()
    assert f"@wllama/wllama@{WLLAMA_VERSION}/" in js and "__WLLAMA_VERSION__" not in js
    assert re.fullmatch(r"\d+\.\d+\.\d+", WLLAMA_VERSION)
    p = json.loads((out / "demo" / "prompt.json").read_text())
    assert p["system"] == prompts.COMPACT_SYSTEM_PROMPT and "{text}" in p["user_template"]
    assert json.loads((out / "demo" / "schema.json").read_text())["title"] == "JobPosting"
    assert "react" in json.loads((out / "demo" / "aliases.json").read_text())["skills"]
    assert len(json.loads((out / "demo" / "examples.json").read_text())) >= 3
    # no model: "demo pending" note, no demo script
    out2 = tmp_path / "site2"
    build_site(".", out2)
    assert "demo/demo.js" not in (out2 / "index.html").read_text()


def test_site_rejects_parts_over_95mb(tmp_path):
    parts = tmp_path / "parts"
    parts.mkdir()
    with open(parts / "big.gguf.part000", "wb") as f:
        f.truncate(96 * 1024 * 1024)
    with pytest.raises(ValueError, match="95 MB"):
        build_site(".", tmp_path / "site", model_dir=parts)


def test_results_table_cpu_route_layout():
    md, _ = build_tables(".")
    synth = md.split("#### Synthetic held-out")[1].split("####")[0]
    assert "Qwen2.5-0.5B LoRA fine-tune (ours), GGUF Q4_K_M" in synth  # measured or pending
    assert "GPT-OSS-120B via Groq" in synth
    assert "| Qwen2.5-7B-Instruct" not in synth and "optional GPU-route rows" in synth
    bench = md.split("#### Size / speed / accuracy trade-off")[1]
    assert "LoRA 0.5B GGUF Q8_0" in bench and "AWQ" not in bench


def test_model_card_renders_without_placeholders():
    card = render_model_card(Path("."))
    assert "{{" not in card and prompts.COMPACT_SYSTEM_PROMPT in card
    assert "Qwen/Qwen2.5-0.5B-Instruct" in card


def test_charts(tmp_path):
    pytest.importorskip("matplotlib")
    from jobx.eval.charts import draw_charts

    root = tmp_path
    (root / "results").mkdir()
    data = {
        "splits": {
            "test_synth": [
                {
                    "name": "A",
                    "status": "measured",
                    "metrics": {"micro_f1": 0.8},
                    "ci95": {"micro_f1": [0.78, 0.82]},
                },
                {"name": "B fine-tune", "status": "measured", "metrics": {"micro_f1": 0.9}, "ci95": None},
                {"name": "C", "status": "pending"},
            ]
        },
        "benchmarks": [{"variant": "Q4", "size_gb": 0.4, "micro_f1": 0.88, "tokens_per_s": 40}],
    }
    (root / "results" / "results.json").write_text(json.dumps(data))
    written = draw_charts(root)
    assert {p.name for p in written} == {"micro_f1_by_approach.png", "quant_tradeoff.png"}


def test_workflows_static():
    ft = yaml.safe_load((WF / "cpu_finetune.yml").read_text())
    job = ft["jobs"]["train"]
    assert job["timeout-minutes"] <= 360 and ft["permissions"]["actions"] == "write"
    text = (WF / "cpu_finetune.yml").read_text()
    assert "gh workflow run cpu_finetune.yml" in text and "gh workflow run cpu_eval_release.yml" in text
    assert "gh release upload" in text and "--clobber" in text and "cron:" in text
    ev = (WF / "cpu_eval_release.yml").read_text()
    assert "gh release create model-v1" in ev and "--notes-file MODEL_CARD.md" in ev and "2000000000" in ev
    pages = (WF / "pages.yml").read_text()
    assert "peaceiris/actions-gh-pages@v4" in pages and "force_orphan: true" in pages
    assert "split -b 50M" in pages and "check_demo.py" in pages and "playwright==" in pages
    for name in ("cpu_baselines.yml", "cpu_finetune.yml", "cpu_eval_release.yml"):
        t = (WF / name).read_text()
        assert "scripts/commit_results.sh" in t and "git push" not in t, name
        assert "secrets." not in t.replace("secrets.GITHUB_TOKEN", ""), f"{name} must not need secrets"
    assert os.access("scripts/commit_results.sh", os.X_OK) and os.access(
        "scripts/build_llama_cpp.sh", os.X_OK
    )
    sh = Path("scripts/commit_results.sh").read_text()
    assert "reset -q --hard" in sh and "for i in 1 2 3 4 5" in sh


def test_guided_schema_requires_every_key():
    from jobx.baselines.guided import guided_schema, require_all

    s = guided_schema()
    assert set(s["required"]) == set(s["properties"]) and s["additionalProperties"] is False
    assert "$ref" not in json.dumps(s) and "$defs" not in s
    sal = next(x for x in s["properties"]["salary"]["anyOf"] if x.get("type") == "object")
    assert set(sal["required"]) == {"min", "max", "currency", "period"}
    assert require_all({"type": "string"}) == {"type": "string"}
