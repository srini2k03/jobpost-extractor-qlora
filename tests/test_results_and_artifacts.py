import json
import re
import shutil
from pathlib import Path

import nbformat
import pytest

from jobx.eval.results_table import END, PENDING, START, build_tables, update_readme
from jobx.prompts import SYSTEM_PROMPT, USER_TEMPLATE, messages, select_few_shot, sft_record, target_json

NOTEBOOKS = sorted(Path("notebooks").glob("*.ipynb"))


def test_results_table_has_measured_and_pending_rows():
    md, data = build_tables(".")
    assert "regex" in md.lower() and PENDING in md
    synth = {r["approach"]: r for r in data["splits"]["test_synth"]}
    assert synth["regex_baseline"]["status"] == "measured"
    assert synth["qwen2.5-1.5b_qlora"]["status"] == "pending"


def test_committed_results_are_current(tmp_path):
    """CI regenerates the table from committed predictions; the committed copy must match."""
    from jobx.eval.run import evaluate_all

    out = tmp_path / "metrics"
    evaluate_all("results/predictions", out)
    for p in out.glob("*.json"):
        assert json.loads(p.read_text()) == json.loads((Path("results/metrics") / p.name).read_text()), p.name
    md, _ = build_tables(".")
    assert Path("results/results_table.md").read_text() == md
    readme = Path("README.md").read_text()
    assert md in readme


def test_update_readme_markers(tmp_path):
    p = tmp_path / "R.md"
    p.write_text(f"a\n{START}\nold\n{END}\nb\n")
    assert update_readme(p, "new\n")
    assert p.read_text() == f"a\n{START}\nnew\n{END}\nb\n"
    q = tmp_path / "Q.md"
    q.write_text("no markers")
    assert not update_readme(q, "x")


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.stem)
def test_notebook_static_checks(path):
    nb = nbformat.read(path, as_version=4)
    nbformat.validate(nb)
    for c in nb.cells:
        if c.cell_type == "code":
            assert not c.outputs and c.execution_count is None, f"{path.name} has outputs committed"
    first = nb.cells[0].source
    url = f"https://colab.research.google.com/github/srini2k03/jobpost-extractor-qlora/blob/main/notebooks/{path.name}"
    assert url in first and "colab-badge.svg" in first
    assert "Expected runtime" in first and "Run all" in first
    code = "\n".join(c.source for c in nb.cells if c.cell_type == "code")
    assert "jobx" in code and "pip install -q -e ." in code
    if path.stem != "06_error_analysis":
        setup = nb.cells[1].source
        pins = re.findall(r"[A-Za-z][\w.\-]*==\S+", setup)
        assert len(pins) >= 4 and all(re.fullmatch(r"[\w.\-]+==\d[\w.]*", p) for p in pins), pins
        assert "bf16" not in code.replace("bf16': False", "") or "fp16" in code
        assert "userdata" in code


def test_notebooks_match_generator(tmp_path):
    import subprocess
    import sys

    shutil.copytree("notebooks", tmp_path / "nb_before")
    subprocess.run([sys.executable, "scripts/build_notebooks.py"], check=True, capture_output=True)
    for p in NOTEBOOKS:
        assert p.read_text() == (tmp_path / "nb_before" / p.name).read_text(), f"{p.name} out of date"


def test_modelfile_matches_prompts():
    mf = Path("ollama/Modelfile").read_text()
    assert f'SYSTEM """{SYSTEM_PROMPT}"""' in mf
    pre, post = USER_TEMPLATE.split("{text}")
    assert pre + "{{ .Content }}" + post in mf


def test_prompt_helpers():
    from jobx.data.build import read_jsonl

    train = read_jsonl("data/processed/train.jsonl")
    shots = select_few_shot(train, k=3)
    assert len(shots) == 3 and len({s["meta"]["family"] for s in shots}) == 3
    assert shots == select_few_shot(train, k=3)
    msgs = messages("hello", shots)
    assert (
        msgs[0]["content"] == SYSTEM_PROMPT and len(msgs) == 8 and msgs[-1]["content"].endswith("</posting>")
    )
    rec = sft_record(train[0])
    assert [m["role"] for m in rec["prompt"]] == ["system", "user"]
    assert json.loads(rec["completion"][0]["content"]) == json.loads(target_json(train[0]["label"]))
