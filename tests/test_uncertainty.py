import json

import numpy as np

from jobx.uncertainty import (
    Calibrator,
    apply_abstention,
    auroc,
    choose_threshold,
    field_features,
    field_logprobs,
    reliability,
    risk_coverage,
    value_spans,
)

OUT = json.dumps({"company": "Kaveri", "role": "Data Engineer", "skills": ["python", "sql"], "salary": None})


def _tokens(text, lp_by_char):
    """Split text into 3-char tokens with a logprob per token (min of the per-char values)."""
    toks = []
    for i in range(0, len(text), 3):
        piece = text[i : i + 3]
        toks.append(
            {
                "token": piece,
                "bytes": list(piece.encode()),
                "logprob": min(lp_by_char(j) for j in range(i, i + len(piece))),
            }
        )
    return toks


def test_value_spans_cover_values_and_list_elements():
    sp = value_spans("Sure! " + OUT)
    text = "Sure! " + OUT
    assert text[slice(*sp["company"])] == '"Kaveri"'
    assert text[slice(*sp["skills"])] == '["python", "sql"]'
    assert text[slice(*sp["skills[1]"])] == '"sql"'
    assert text[slice(*sp["salary"])] == "null"
    assert value_spans("no json here") == {}


def test_field_logprobs_attribute_tokens_to_fields():
    s, e = value_spans(OUT)["role"]
    toks = _tokens(OUT, lambda j: -2.0 if s <= j < e else -0.01)
    lp = field_logprobs(toks, OUT)
    assert lp["role"]["min"] == -2.0 and lp["company"]["min"] > -2.0
    assert lp["skills[0]"]["n"] >= 1


def test_field_logprobs_handle_multibyte_characters_split_across_tokens():
    out = json.dumps({"company": "Café ₹", "role": None}, ensure_ascii=False)
    raw = out.encode()
    toks = [{"token": "?", "bytes": list(raw[i : i + 2]), "logprob": -0.5} for i in range(0, len(raw), 2)]
    lp = field_logprobs(toks, out)
    assert set(lp) == {"company", "role"} and lp["company"]["mean"] == -0.5


def test_calibrator_learns_a_monotone_map_and_round_trips():
    rng = np.random.default_rng(0)
    n = 600
    mean_lp = -rng.exponential(0.5, n)
    y = (rng.random(n) < np.exp(mean_lp)).astype(float)
    X = np.array([field_features({"mean": m, "min": m * 1.5}, "supported", False) for m in mean_lp])
    fields = ["company"] * n
    cal = Calibrator().fit(X, y, fields)
    p = cal.predict(X, fields)
    assert auroc(p, y) > 0.6
    overconfident = np.exp(mean_lp / 4)  # raw probabilities that claim too much
    assert reliability(p, y)["ece"] < 0.1 < reliability(overconfident, y)["ece"]
    again = Calibrator.from_dict(json.loads(json.dumps(cal.to_dict()))).predict(X, fields)
    assert np.allclose(p, again)


def test_reliability_and_risk_coverage_basics():
    p = np.array([0.9, 0.8, 0.3, 0.2])
    y = np.array([1, 1, 0, 1])
    r = reliability(p, y, bins=5)
    assert r["n"] == 4 and 0 <= r["ece"] <= 1 and r["auroc"] == 0.6667
    rc = risk_coverage(p, y, [0.0, 0.5])
    assert rc[0]["coverage"] == 1.0 and rc[1]["coverage"] == 0.5 and rc[1]["selective_accuracy"] == 1.0
    assert choose_threshold(p, y, 1.0) == 0.8
    assert auroc(np.array([0.5, 0.5]), np.array([1, 0])) == 0.5


def test_abstention_modes_keep_withheld_values():
    data = {"company": "X", "skills": ["a"], "role": "Dev"}
    conf = {"company": 0.2, "skills": 0.4, "role": 0.95}
    out, unc, held = apply_abstention(data, conf, threshold=0.5, mode="null")
    assert out["company"] is None and out["skills"] == [] and out["role"] == "Dev"
    assert unc == ["company", "skills"] and held == {"company": "X", "skills": ["a"]}
    out, unc, held = apply_abstention(data, conf, threshold=0.5, mode="flag")
    assert out == data and unc == ["company", "skills"] and held == {}
    assert apply_abstention(data, conf, threshold=0.5, mode="off")[1] == []
