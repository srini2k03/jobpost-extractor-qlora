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


TEXT = (
    "Hiring Python Developer @ Kaveri Analytics Pvt Ltd\nExp: 4-7 yrs | CTC 18-24 LPA\n"
    "Loc: Bangalore / Pune (Hybrid)\nMust know Django, AWS, PostgreSQL, Docker\nB.Tech/B.E. preferred."
)


def test_extract_returns_evidence_spans_and_flags_unsupported_values():
    out = {**gold_example(), "skills": [*gold_example()["skills"], "kubernetes"]}
    c = TestClient(create_app(MockLLM([json.dumps(out)])))
    body = c.post("/extract", json={"text": TEXT}).json()
    ev = {(e["field"], json.dumps(e["value"])): e for e in body["evidence"]}
    assert ev[("skills", '"kubernetes"')]["status"] == "unsupported"
    loc = ev[("locations", '"Bengaluru"')]
    assert (
        loc["status"] == "supported"
        and TEXT[loc["spans"][0]["start"] : loc["spans"][0]["end"]] == "Bangalore"
    )
    assert body["evidence_summary"]["unsupported"] >= 1
    assert body["confidence_source"] == "evidence" and body["confidence"]["skills"] == 0.0


def test_abstention_null_withholds_unsupported_fields_but_returns_them():
    out = {**gold_example(), "work_mode": "remote"}
    c = TestClient(create_app(MockLLM([json.dumps(out)])))
    body = c.post("/extract", json={"text": TEXT, "abstain": "null"}).json()
    assert "work_mode" in body["uncertain_fields"] and body["data"]["work_mode"] is None
    assert body["withheld"]["work_mode"] == "remote"
    flag = (
        TestClient(create_app(MockLLM([json.dumps(out)])))
        .post("/extract", json={"text": TEXT, "abstain": "flag"})
        .json()
    )
    assert flag["data"]["work_mode"] == "remote" and "work_mode" in flag["uncertain_fields"]


def test_calibrated_confidence_with_logprobs(tmp_path):
    from jobx.serve.config import Settings
    from jobx.uncertainty import FEATURES

    cal = {
        "approach": "test",
        "threshold": 0.9,
        "features": list(FEATURES),
        "weights": [4.0, 0.0, 0.0, 1.0, 0.0, -2.0, 0.0],
        "field_bias": {},
        "l2": 1.0,
        "n_train": 10,
    }
    p = tmp_path / "cal.json"
    p.write_text(json.dumps(cal))
    app = create_app(MockLLM([GOOD], logprob=-0.01), settings=Settings(calibrator_path=p))
    body = TestClient(app).post("/extract", json={"text": TEXT, "abstain": "flag"}).json()
    assert body["confidence_source"] == "calibrated" and body["threshold"] == 0.9
    assert set(body["confidence"]) == set(gold_example()) and all(
        0 <= v <= 1 for v in body["confidence"].values()
    )
    assert TestClient(app).get("/model").json()["calibration"]["threshold"] == 0.9


def test_batch_validate_evidence_and_limits():
    from jobx.serve.config import Settings

    app = create_app(MockLLM(lambda m: GOOD), settings=Settings(max_batch=2, rate_limit_per_min=0))
    c = TestClient(app)
    r = c.post(
        "/extract/batch", json={"items": [{"id": "a", "text": TEXT}, {"text": "x" * (MAX_CHARS + 5)}]}
    ).json()
    assert r["results"][0]["id"] == "a" and r["results"][0]["valid"] and "error" in r["results"][1]
    assert c.post("/extract/batch", json={"items": [{"text": "a"}] * 3}).status_code == 413
    v = c.post("/validate", json={"data": {**gold_example(), "work_mode": "WFH"}}).json()
    assert not v["valid"] and v["errors"][0]["loc"] == "work_mode"
    v = c.post("/validate", json={"data": gold_example(), "text": TEXT}).json()
    assert v["valid"] and v["evidence_summary"]["n_values"] > 0
    e = c.post("/evidence", json={"text": TEXT, "data": {"skills": ["django", "rust"]}}).json()
    assert [x["status"] for x in e["evidence"]] == ["supported", "unsupported"]
    m = c.get("/model").json()
    assert m["limits"]["max_batch"] == 2 and m["prompt"] == "compact"


def test_rate_limit_and_request_ids():
    from jobx.serve.app import RateLimiter
    from jobx.serve.config import Settings

    rl = RateLimiter(2)
    assert rl.allow("k", 0) and rl.allow("k", 1) and not rl.allow("k", 2) and rl.allow("k", 61)
    c = TestClient(create_app(MockLLM(lambda m: GOOD), settings=Settings(rate_limit_per_min=1)))
    first = c.post("/extract", json={"text": TEXT})
    assert first.status_code == 200 and first.headers["x-request-id"]
    assert c.post("/extract", json={"text": TEXT}).status_code == 429
    assert c.get("/health").status_code == 200  # reads are not rate-limited


def test_logs_never_contain_the_posting(caplog):
    import logging

    secret = "Call HR on 9876543210 at secret-company"
    with caplog.at_level(logging.INFO, logger="jobx.serve"):
        TestClient(create_app(MockLLM([GOOD]))).post("/extract", json={"text": secret})
    assert caplog.records and all(
        secret not in r.getMessage() and secret not in str(getattr(r, "fields", "")) for r in caplog.records
    )
