"""Local-GPU path, CPU-only: profiles load and validate, CLI wiring resolves, the check degrades gracefully."""

import argparse
import re
from pathlib import Path

import pytest
import yaml

from jobx import pipeline
from jobx.local import __main__ as cli
from jobx.local import check
from jobx.train.config import load_train_config


def test_local_train_profile_inherits_and_overrides():
    t4 = load_train_config("configs/train/qwen2.5-1.5b-qlora.yaml")
    loc = load_train_config("configs/train/local_rtx4080.yaml")
    lt, tt = loc["training"], t4["training"]
    assert lt["bf16"] is True and lt["fp16"] is False and lt["tf32"] is True
    assert loc["quantization"]["bnb_4bit_compute_dtype"] == "bfloat16"
    assert lt["per_device_train_batch_size"] == 8 and lt["gradient_accumulation_steps"] == 2
    eff = lambda t: t["per_device_train_batch_size"] * t["gradient_accumulation_steps"]  # noqa: E731
    assert eff(lt) == eff(tt) == 16
    assert lt["max_length"] == 1024 and lt["gradient_checkpointing"] and lt["optim"] == "paged_adamw_8bit"
    assert 0 < lt["dataloader_num_workers"] <= 4 and lt["resume"] is True
    # the recipe itself is unchanged
    assert loc["lora"] == t4["lora"] and lt["learning_rate"] == tt["learning_rate"]
    assert loc["base_model"] == t4["base_model"] and loc["data"] == t4["data"]
    assert lt["output_dir"] != tt["output_dir"] and loc["run_name"] != t4["run_name"]


@pytest.mark.parametrize("name", ["colab_t4", "local_rtx4080"])
def test_gpu_profiles_validate(name):
    prof = pipeline.load_profile(name)
    assert prof.name == name and prof.dtype == "auto"
    assert Path(prof.train_config).exists()
    load_train_config(prof.train_config)
    assert prof.baselines["guided_max"] == 200  # identical subset on both paths -> comparable rows
    assert prof.paraphrase["teacher"] == "Qwen/Qwen2.5-7B-Instruct"


def test_profile_validation_errors(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        yaml.safe_dump({"name": "x", "paraphrase": {"n": 1}, "baselines": {}, "eval": {}, "awq": {}})
    )
    with pytest.raises(ValueError, match="missing"):
        pipeline.load_profile(bad)
    d = yaml.safe_load(Path("configs/gpu/local_rtx4080.yaml").read_text())
    d["dtype"] = "int4"
    bad.write_text(yaml.safe_dump(d))
    with pytest.raises(ValueError, match="dtype"):
        pipeline.load_profile(bad)


def test_cli_arguments_resolve():
    p = cli.parser()
    a = p.parse_args(["baselines", "--families", "qwen1.5b", "--push"])
    assert a.step == "baselines" and a.profile == cli.DEFAULT_PROFILE and a.push
    for step in ("check", "paraphrase", "baselines", "train", "eval", "awq", "push", "all"):
        assert p.parse_args([step]).step == step
        assert step in cli.EXPECTED_RUNTIME
    with pytest.raises(SystemExit):
        p.parse_args(["nonsense"])
    with pytest.raises(SystemExit, match="unknown baseline families"):
        cli.run_step("baselines", p.parse_args(["baselines", "--families", "gpt5"]))


def test_run_all_order_and_summary(capsys):
    seen = []

    def fake(step: str, a: argparse.Namespace) -> None:
        seen.append(step)
        if step == "baselines":
            raise RuntimeError("boom")

    summary = cli.run_all(cli.parser().parse_args(["all"]), runner=fake)
    assert seen == ["paraphrase", "baselines", "train", "eval"]
    assert [s["status"] == "ok" for s in summary] == [True, False, True, True]
    out = capsys.readouterr().out
    assert "=== summary ===" in out and "FAILED: RuntimeError: boom" in out


def test_makefile_targets_call_the_cli():
    mk = Path("Makefile").read_text()
    for target, step in [
        ("local-check", "check"),
        ("local-paraphrase", "paraphrase"),
        ("local-baselines", "baselines"),
        ("local-train", "train"),
        ("local-eval", "eval"),
        ("local-awq", "awq"),
        ("local-push", "push"),
        ("local-all", "all"),
    ]:
        body = mk.split(f"\n{target}:", 1)[1].split("\n\n", 1)[0]
        assert re.search(
            rf"jobx\.local {step}\b|\$\(LOCAL\) {step}\b|\$\(AWQ_PY\) -m jobx\.local {step}\b", body
        ), target
    assert "requirements/local-gpu.txt" in mk and "requirements/local-awq.txt" in mk


def test_requirements_pins_are_consistent_with_notebooks():
    pins = check.pinned_versions(Path("requirements/local-gpu.txt"))
    assert pins["torch"] == "2.8.0" and "cu128" in Path("requirements/local-gpu.txt").read_text()
    from scripts_build_notebooks import GPU_PINS  # type: ignore[import-not-found]

    nb = dict(t.split("==") for t in GPU_PINS.split())
    for k, v in nb.items():
        assert pins[k.lower().replace("_", "-")] == v, k
    awq = check.pinned_versions(Path("requirements/local-awq.txt"))
    assert awq["autoawq"] == "0.2.9" and awq["transformers"] == "4.51.3"


def test_check_handles_no_gpu_environment(monkeypatch, tmp_path):
    """This CI machine has no torch/GPU: every check still returns, torch-related ones FAIL, exit code 1."""
    req = tmp_path / "req.txt"
    req.write_text("--extra-index-url https://example.invalid\npydantic==0.0.1\nnumpy>=1\n")
    checks = check.run_checks(req)
    by = {c.name: c for c in checks}
    assert by["package versions"].status == check.FAIL and "pydantic" in by["package versions"].detail
    for name in ("cuda", "bf16", "bitsandbytes 4-bit"):
        assert by[name].status == check.FAIL
    assert by["HF login"].status == check.FAIL  # conftest clears HF_TOKEN
    assert {c.status for c in checks} <= {check.PASS, check.WARN, check.FAIL}
    table = check.render(checks)
    assert "RESULT: FAIL" in table and "bitsandbytes 4-bit" in table
    assert check.main(["--requirements", str(req), "--no-bnb"]) == 1


def test_check_passes_versions_when_installed(tmp_path):
    import importlib.metadata as md

    req = tmp_path / "req.txt"
    req.write_text(f"pydantic=={md.version('pydantic')}\nPyYAML=={md.version('PyYAML')}\n")
    assert check.check_versions(req).status == check.PASS


def test_push_plan_matches_notebook_paths(tmp_path):
    root = tmp_path
    for rel in (
        "results/predictions/a__test_synth.jsonl",
        "results/metrics/a__test_synth.json",
        "results/benchmarks/local_rtx4080.json",
        "results/training/run/log_history.json",
        "data/paraphrased/paraphrased_train.jsonl",
        "data/paraphrased/raw_generations.jsonl",
        "results/training/README.md",
        "outputs/merged/config.json",
    ):
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text("{}")
    (root / "configs").symlink_to(Path("configs").resolve())
    plan = pipeline.push_outputs(pipeline.load_profile("local_rtx4080", root=root), root=root, dry_run=True)
    files = set(plan["artifact_files"])
    assert (
        "results/predictions/a__test_synth.jsonl" in files
        and "results/training/run/log_history.json" in files
    )
    assert "data/paraphrased/raw_generations.jsonl" not in files  # intermediate, never pushed
    assert "results/training/README.md" not in files
    assert plan["model_folders"] == {"merged": "outputs/merged"}
