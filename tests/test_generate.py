import json
import random
from pathlib import Path

from jobx import vocab
from jobx.data.generate import GenConfig, generate
from jobx.data.noise import DISTRACTORS
from jobx.data.templates import FAMILIES, PERKS, RESPONSIBILITIES
from jobx.data.validate import check_recoverable
from jobx.normalize.salary import find_salaries
from jobx.normalize.text import matchers
from jobx.schema import JobPosting


def test_labels_are_schema_valid_and_recoverable(small_generated):
    """Labels by construction: every gold value can be found in the rendered, noised text."""
    failures = []
    for ex in small_generated:
        JobPosting.model_validate(ex["label"])
        res = check_recoverable(ex["text"], ex["label"])
        if not res.ok:
            failures.append((ex["id"], res.failures))
    assert not failures, failures[:5]


def test_generation_is_deterministic():
    import yaml

    cfg = yaml.safe_load(Path("configs/data.yaml").read_text())
    cfg["n_examples"] = 30
    a = generate(GenConfig.from_dict(cfg))
    b = generate(GenConfig.from_dict(cfg))
    assert json.dumps(a) == json.dumps(b)


def test_all_families_used_and_configured(small_generated):
    import yaml

    cfg = yaml.safe_load(Path("configs/data.yaml").read_text())
    configured = {f for fams in cfg["families"].values() for f in fams}
    assert configured == set(FAMILIES)
    assert {e["meta"]["family"] for e in small_generated} == configured


def test_absent_fields_really_absent(small_generated):
    """Null gold fields must not be stated: no salary mention when salary is null (distractors aside)."""
    for ex in small_generated:
        if ex["label"]["salary"] is None and "distractors" not in ex["meta"]["noise_ops"]:
            assert find_salaries(ex["text"]) == [], ex["text"]
        if not ex["label"]["locations"] and not ex["meta"]["hq_trap"]:
            assert matchers()["locations"].find(ex["text"]) == [], ex["text"]


def test_boilerplate_has_no_field_values():
    """Generic lines must not mention skills/cities/degrees, or labels would be wrong."""
    rng = random.Random(0)
    lines = (
        list(RESPONSIBILITIES) + list(PERKS) + list(vocab.HINGLISH_FRAGMENTS) + [d(rng) for d in DISTRACTORS]
    )
    m = matchers()
    for line in lines:
        for field in ("skills", "locations", "education", "work_mode", "employment_type"):
            assert m[field].find(line) == [], (field, line)


def test_distractors_not_parsed_as_salary():
    rng = random.Random(1)
    for d in DISTRACTORS:
        for _ in range(5):
            assert find_salaries(d(rng)) == [], d(rng)


def test_diversity(small_generated):
    texts = {e["text"] for e in small_generated}
    assert len(texts) == len(small_generated)
    salaries = [e["label"]["salary"] for e in small_generated if e["label"]["salary"]]
    assert {s["period"] for s in salaries} == {"year", "month"}
    assert any(s["min"] is None for s in salaries) and any(s["max"] is None for s in salaries)
    assert any(e["label"]["company"] is None for e in small_generated)
