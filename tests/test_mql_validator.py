"""Read-only MQL validator (eval_kit.validate_mql): `$unionWith` target checks, short and long
form, at any nesting depth. Offline -- no Atlas, no model calls."""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/evaluation"))
from eval_kit import validate_mql  # noqa: E402

ALLOWED = {"dog_kennels__Owners", "dog_kennels__Professionals", "dog_kennels__Dogs"}
DATASET = ROOT / "data/processed/spider1_nl_mql.jsonl"
COLLECTIONS = ROOT / "data/interim/spider1/atlas_collections.json"


def check(pipeline, allowed=ALLOWED, coll="dog_kennels__Professionals"):
    return validate_mql({"method": "aggregate", "collection": coll, "pipeline": pipeline}, allowed)


def lookup(pipeline):
    return {"$lookup": {"from": "dog_kennels__Dogs", "pipeline": pipeline, "as": "j"}}


@pytest.mark.parametrize("pipeline", [
    [{"$unionWith": "dog_kennels__Owners"}],
    [{"$unionWith": {"coll": "dog_kennels__Owners"}}],
    [{"$unionWith": {"coll": "dog_kennels__Owners", "pipeline": [{"$project": {"first_name": 1}}]}}],
    [{"$facet": {"a": [{"$unionWith": "dog_kennels__Owners"}]}}],
    [lookup([{"$unionWith": "dog_kennels__Owners"}])],
    [{"$unionWith": {"coll": "dog_kennels__Owners", "pipeline": [{"$unionWith": "dog_kennels__Dogs"}]}}],
])
def test_union_with_permitted_target_is_accepted(pipeline):
    ok, why, norm = check(pipeline)
    assert ok, why
    assert norm["pipeline"] == pipeline


@pytest.mark.parametrize("pipeline", [
    [{"$unionWith": "pets_1__Pets"}],                                   # other database
    [{"$unionWith": "system.users"}],
    [{"$unionWith": {"coll": "pets_1__Pets"}}],
    [{"$facet": {"a": [{"$unionWith": "pets_1__Pets"}]}}],              # nested in $facet
    [lookup([{"$unionWith": "pets_1__Pets"}])],                         # nested in $lookup
    [{"$unionWith": {"coll": "dog_kennels__Owners",                     # nested in $unionWith
                     "pipeline": [{"$unionWith": "pets_1__Pets"}]}}],
])
def test_union_with_unauthorized_target_is_rejected(pipeline):
    ok, why, norm = check(pipeline)
    assert not ok and norm is None
    assert "pets_1__Pets" in why or "system.users" in why


@pytest.mark.parametrize("value", [
    "", "   ", None, 5, ["dog_kennels__Owners"], True,
    {}, {"coll": ""}, {"coll": "  "}, {"coll": 5}, {"coll": None},
    {"pipeline": [{"$documents": [{"a": 1}]}]},                         # collection-less form
    {"coll": "dog_kennels__Owners", "db": "admin"},                     # cross-database
    {"coll": "dog_kennels__Owners", "pipeline": {"$match": {}}},
])
def test_union_with_malformed_value_is_rejected(value):
    ok, _, norm = check([{"$unionWith": value}])
    assert not ok and norm is None


def test_malformed_union_with_is_rejected_even_without_an_allow_list():
    assert not check([{"$unionWith": ""}], allowed=set())[0]
    assert not check([{"$unionWith": {"coll": 1}}], allowed=None)[0]


def test_existing_checks_are_unchanged():
    assert check([{"$match": {}}])[0]
    assert check([{"$lookup": {"from": "dog_kennels__Dogs", "localField": "a",
                               "foreignField": "b", "as": "j"}}])[0]
    assert not check([{"$lookup": {"from": "pets_1__Pets", "localField": "a",
                                   "foreignField": "b", "as": "j"}}])[0]
    assert not check([{"$out": "dog_kennels__Owners"}])[0]
    assert not check([{"$match": {}}], coll="pets_1__Pets")[0]
    # no allow-list: reference checks stay disabled (legacy behaviour)
    assert check([{"$unionWith": "anything"}], allowed=set())[0]


@pytest.mark.skipif(not (DATASET.exists() and COLLECTIONS.exists()), reason="dataset not built")
def test_all_gold_mql_pass_the_validator():
    colls = json.loads(COLLECTIONS.read_text(encoding="utf-8")).keys()
    n = 0
    for line in DATASET.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        allowed = {c for c in colls if c.startswith(r["db_id"] + "__")}
        ok, why, _ = validate_mql({"method": "aggregate", **r["gold_mql"]}, allowed)
        assert ok, (r["id"], why)
        n += 1
    assert n == 279
