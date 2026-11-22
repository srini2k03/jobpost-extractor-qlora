"""Phase 4: quality per quantization rows (vLLM / llama.cpp on Modal) and their place in the results table."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import yaml

from jobx import prompts
from jobx.baselines.openai_compat_runner import build_request, parser
from jobx.bench import plan
from jobx.eval.results_table import build_tables

ROOT = Path(__file__).resolve().parents[1]


def test_every_quality_pass_has_a_results_row() -> None:
    planned = {
        a["id"]: a for a in yaml.safe_load((ROOT / "configs/approaches.yaml").read_text())["approaches"]
    }
    ids = [i for c in plan.expand(plan.load_matrix()) for i in c.quality_approach_ids()]
    assert len(ids) == len(set(ids)) == 10
    for i in ids:
        a = planned[i]
        assert a["route"] == "vllm" and a["splits"] == ["test_synth"]
        assert a["decoding"] == ("guided" if "-guided_" in i else "free")
    assert "vllm-qwen2.5-0.5b-lora_zero_shot" in ids  # compact prompt, no shots


def test_runner_compact_system_prompt_and_meta_flags() -> None:
    req = build_request("m", "posting", [], mode="none", max_tokens=384, system="compact")
    assert req["messages"][0]["content"] == prompts.COMPACT_SYSTEM_PROMPT and "response_format" not in req
    full = build_request("m", "posting", [], mode="json_schema", max_tokens=384)
    assert (
        full["messages"][0]["content"] == prompts.SYSTEM_PROMPT
        and full["response_format"]["type"] == "json_schema"
    )
    a = parser().parse_args(["--system", "compact", "--hardware", "NVIDIA L4 (Modal)", "--commit", "abc"])
    assert (a.system, a.hardware, a.commit) == ("compact", "NVIDIA L4 (Modal)", "abc")


def test_pending_serving_rows_are_summarised_not_listed() -> None:
    md, data = build_tables(ROOT)
    synth = md.split("#### Synthetic held-out")[1].split("####")[0]
    real = md.split("#### Real postings")[1].split("####")[0]
    assert "serving-benchmark quality rows" in synth and "vLLM BF16 |" not in synth
    assert "serving-benchmark quality rows" not in real  # planned for test_synth only
    assert "vllm-qwen2.5-1.5b-awq_few_shot" not in {r["approach"] for r in data["splits"]["test_real"]}
    pending = {r["approach"]: r["status"] for r in data["splits"]["test_synth"]}
    assert pending["vllm-qwen2.5-1.5b-awq_few_shot"] == "pending"


def test_measured_serving_row_appears_in_table(tmp_path: Path) -> None:
    root = tmp_path
    shutil.copytree(ROOT / "configs", root / "configs")
    (root / "results" / "metrics").mkdir(parents=True)
    (root / "results" / "benchmarks").mkdir(parents=True)
    src = ROOT / "results/metrics/qwen2.5-1.5b-cpu_few_shot__test_synth.json"
    d = json.loads(src.read_text())
    d["approach"] = "vllm-qwen2.5-1.5b-awq_few_shot"
    d["meta"] = {**(d.get("meta") or {}), "hardware": "NVIDIA L4 (Modal), vllm 0.33.0", "date": "2026-12-01"}
    (root / "results/metrics/vllm-qwen2.5-1.5b-awq_few_shot__test_synth.json").write_text(json.dumps(d))
    md, _ = build_tables(root)
    synth = md.split("#### Synthetic held-out")[1].split("####")[0]
    assert "| Qwen2.5-1.5B-Instruct 3-shot, vLLM AWQ int4 | L4 (Modal), vLLM | 471 |" in synth
    assert "_9 serving-benchmark quality rows" in synth


def test_quality_only_selection_skips_load() -> None:
    m = plan.load_matrix()
    sel = plan.plan_selection(m, config="all", quality_only=True)
    assert {c.name for c in sel.configs} == {c.name for c in plan.expand(m) if c.quality}
    assert sel.skip_load and all(e["load_min"] == 0 for e in sel.decision.estimates)
    assert sel.decision.total_minutes < 90
