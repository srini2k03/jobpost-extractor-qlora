import json

from fastapi.testclient import TestClient

from conftest import gold_example
from jobx.serve.app import MAX_CHARS, create_app
from jobx.serve.extractor import Extractor
from jobx.serve.llm import MockLLM, RegexLLM

GOOD = json.dumps(gold_example())


def test_extract_valid_first_try():
    llm = MockLLM([GOOD])
    c = TestClient(create_app(llm))
    r = c.post("/extract", json={"text": "some posting"})
    assert r.status_code == 200
    body = r.json()
    assert body["valid"] and body["attempts"] == 1 and body["data"]["company"] == "Kaveri Analytics"
    assert "<posting>\nsome posting\n</posting>" in llm.calls[0][-1]["content"]


def test_retry_on_invalid_json_then_success():
    llm = MockLLM(["{not json", json.dumps({**gold_example(), "work_mode": "WFH"}), GOOD])
    c = TestClient(create_app(llm, max_retries=2))
    body = c.post("/extract", json={"text": "posting"}).json()
    assert body["valid"] and body["attempts"] == 3 and len(body["errors"]) == 2
    retry_msg = llm.calls[1][-1]["content"]
    assert "not valid" in retry_msg and llm.calls[1][-2]["role"] == "assistant"


def test_retry_disabled_returns_invalid():
    c = TestClient(create_app(MockLLM(["garbage"])))
    body = c.post("/extract", json={"text": "posting", "retry": False}).json()
    assert not body["valid"] and body["attempts"] == 1 and body["data"] is None


def test_guided_flag_passes_schema():
    seen = {}

    def fn(msgs):
        return GOOD

    class Spy(MockLLM):
        def chat(self, messages, *, schema=None, max_tokens=384):
            seen["schema"] = schema
            return super().chat(messages, schema=schema, max_tokens=max_tokens)

    c = TestClient(create_app(Spy(fn)))
    c.post("/extract", json={"text": "posting", "guided": True})
    assert seen["schema"]["title"] == "JobPosting"


def test_health_schema_and_limits():
    c = TestClient(create_app(MockLLM([GOOD])))
    assert c.get("/health").json()["status"] == "ok"
    assert c.get("/schema").json()["title"] == "JobPosting"
    assert c.post("/extract", json={"text": "x" * (MAX_CHARS + 1)}).status_code == 413
    assert c.post("/extract", json={"text": ""}).status_code == 422


def test_regex_backend_end_to_end():
    r = Extractor(RegexLLM()).extract("Role: Telecaller\nSalary: 15k-20k/month\nLocation: Chennai")
    assert r.valid and r.data["salary"]["period"] == "month" and r.data["locations"] == ["Chennai"]


def test_env_backend_regex(monkeypatch):
    monkeypatch.setenv("JOBX_BACKEND", "regex")
    c = TestClient(create_app())
    body = c.post("/extract", json={"text": "Role: Telecaller\nLocation: Chennai"}).json()
    assert body["valid"] and body["backend"] == "regex-baseline"
