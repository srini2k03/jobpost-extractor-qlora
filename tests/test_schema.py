import json
from pathlib import Path

import jsonschema
import pytest
from pydantic import ValidationError

from conftest import gold_example
from jobx.schema import FIELD_ORDER, JobPosting, json_schema, parse_output


def test_committed_schema_matches_models():
    committed = json.loads(Path("schema/jobposting.schema.json").read_text())
    assert committed == json_schema(), "run `jobx schema` to refresh schema/jobposting.schema.json"


def test_gold_validates_with_pydantic_and_jsonschema():
    g = gold_example()
    JobPosting.model_validate(g)
    jsonschema.validate(g, json_schema())


def test_key_order_and_int_rendering():
    d = JobPosting.model_validate(gold_example()).to_json_dict()
    assert tuple(d) == FIELD_ORDER
    assert isinstance(d["min_experience_years"], int)
    assert isinstance(d["salary"]["min"], int)


@pytest.mark.parametrize(
    "patch",
    [
        {"extra": 1},
        {"seniority": "ninja"},
        {"min_experience_years": 9, "max_experience_years": 3},
        {"salary": {"min": 5, "max": 1, "currency": "INR", "period": "year"}},
        {"salary": {"min": None, "max": None, "currency": "INR", "period": "year"}},
        {"work_mode": "wfh"},
    ],
)
def test_invalid_records_rejected(patch):
    with pytest.raises(ValidationError):
        JobPosting.model_validate({**gold_example(), **patch})


def test_empty_posting_is_all_null():
    d = JobPosting().to_json_dict()
    assert d["skills"] == [] and d["locations"] == [] and d["salary"] is None and d["company"] is None


def test_parse_output_variants():
    good = json.dumps(gold_example())
    assert parse_output(good).schema_ok
    fenced = parse_output(f"Sure! Here it is:\n```json\n{good}\n```\nHope this helps")
    assert fenced.json_ok and fenced.schema_ok
    trunc = parse_output(good[:60])
    assert not trunc.json_ok and trunc.truncated
    bad_enum = parse_output(json.dumps({**gold_example(), "work_mode": "WFH"}))
    assert bad_enum.json_ok and not bad_enum.schema_ok and bad_enum.data["company"] == "Kaveri Analytics"
    assert not parse_output("no json here").json_ok
    assert not parse_output("[1, 2]").json_ok
    braces_in_string = parse_output('{"company": "A {weird} name", "role": null}')
    assert braces_in_string.json_ok and braces_in_string.data["company"] == "A {weird} name"
