"""bench.yml must never run (or bill) without the Modal secrets; `jobx bench plan` feeds it."""

from __future__ import annotations

from pathlib import Path

import yaml
from typer.testing import CliRunner

from jobx.cli import app

WF = Path(".github/workflows/bench.yml")


def test_bench_workflow_skips_cleanly_without_secrets() -> None:
    wf = yaml.safe_load(WF.read_text())
    on = wf[True]  # YAML 1.1 reads the `on:` key as True
    assert on["push"]["paths"] == [".github/bench-request.json"] and "workflow_dispatch" in on
    assert wf["permissions"] == {"contents": "write"}
    job = wf["jobs"]["bench"]
    assert job["env"]["MODAL_TOKEN_ID"] == "${{ secrets.MODAL_TOKEN_ID }}"
    first, *rest = job["steps"]
    assert "::notice::" in first["run"] and "ok=false" in first["run"]
    assert rest and all("steps.secrets.outputs.ok == 'true'" in s.get("if", "") for s in rest)
    run = next(s for s in rest if s.get("name") == "Run on Modal")["run"]
    assert "modal run -m jobx.bench.modal_app" in run
    commit = next(s for s in rest if s.get("name") == "Commit results")["run"]
    assert "git pull --rebase" in commit and "results/bench" in commit


def test_plan_cli_outputs_and_refusal(tmp_path: Path) -> None:
    runner = CliRunner()
    out = tmp_path / "gh_output"
    req = tmp_path / "req.json"
    req.write_text('{"configs": ["q15-bf16", "q15-awq"], "levels": {"duration_s": 30}, "skip_quality": true}')
    res = runner.invoke(app, ["bench", "plan", "--request", str(req), "--github-output", str(out)])
    assert res.exit_code == 0, res.output
    kv = dict(line.split("=", 1) for line in out.read_text().splitlines())
    assert kv["configs"] == "q15-bf16,q15-awq" and kv["levels"] == '{"duration_s":30}'
    assert (
        kv["skip_quality"] == "true"
        and kv["yes"] == "false"
        and kv["ok"] == "true"
        and float(kv["minutes"]) > 0
    )
    res = runner.invoke(app, ["bench", "plan", "--config", "all", "--max-gpu-minutes", "30"])
    assert res.exit_code == 2 and "--yes" in res.output
    res = runner.invoke(app, ["bench", "plan", "--config", "all", "--max-gpu-minutes", "30", "--yes"])
    assert res.exit_code == 0
    res = runner.invoke(app, ["bench", "plan"])  # default: the whole matrix within the default cap
    assert res.exit_code == 0 and "**total**" in res.output
