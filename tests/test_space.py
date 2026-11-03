import json
import sys
from pathlib import Path

import pytest

gr = pytest.importorskip("gradio")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "space"))

from conftest import gold_example  # noqa: E402
from jobx.serve.extractor import Extractor  # noqa: E402
from jobx.serve.highlight import highlighted  # noqa: E402
from jobx.serve.llm import MockLLM  # noqa: E402


def test_space_builds_and_predicts_with_mock_model():
    import app as space_app

    demo = space_app.build_demo(MockLLM(lambda m: json.dumps(gold_example())), banner="test")
    assert isinstance(demo, gr.Blocks)
    predict = space_app.make_predict(Extractor(MockLLM(lambda m: json.dumps(gold_example()))))
    text = "Senior Data Engineer at Kaveri Analytics, Bangalore/Pune, Hybrid, 5-8 yrs, 18-24 LPA, Python Spark AWS"
    data, hl, status = predict(text)
    assert data["company"] == "Kaveri Analytics" and status.startswith("valid")
    labels = {lab for _, lab in hl if lab}
    assert {"company", "skill", "location", "salary", "experience", "work_mode"} <= labels
    assert "".join(seg for seg, _ in hl) == text


def test_space_examples_and_fallback(monkeypatch):
    import app as space_app

    monkeypatch.setenv("JOBX_BACKEND", "regex")
    llm, banner = space_app.default_llm()
    assert llm.name == "regex-baseline" and "fallback" in banner.lower()
    assert len(space_app.load_examples()) >= 3


def test_highlight_no_prediction():
    assert highlighted("abc", None) == [("abc", None)]
