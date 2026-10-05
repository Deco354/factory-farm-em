"""Tests for the v2 generator (generate.py).

Replaces the v1 tests: assign_hash_ids no longer exists, the batch shape is now
{"parameters": [...], "items": [...]}, and consolidation no longer stamps
passed_deduplication=True.
"""

import copy
import json
from pathlib import Path

import pytest

from fragile_compassion.sft_document_generation import generate as g
from fragile_compassion.sft_document_generation.generate import (
    build_user_payload,
    canon_num,
    check_item,
    check_parameter,
    check_run_level,
    clean_markdown_json,
    consolidate_output_directory,
    expand_records,
    expected_value,
    load_personas,
    numbers_in,
    parse_batch_response,
    parse_chunk_filename,
    process_chunk,
    sample_personas,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

CHUNK = (
    "Stocking density should be 30-39 kg/m2 depending on house type. "
    "Higher densities reduce cost per kilogram produced. "
    "Lower densities reduce footpad dermatitis and leg problems. "
    "At densities above 35 kg/m2, minimum ventilation must be increased to 0.5 m3/h per kg. "
    "Mortality increases at higher stocking density."
)

PROMPT = (
    "I run four broiler houses on contract in north Georgia and settlement is ranked on "
    "feed conversion. What stocking density should I plan for on my next placement, and "
    "what do I need to watch?"
)

CRITIQUE_PROMPT = (
    "We are planning to place birds at 34.5 kg/m2 in our three older curtain-sided houses "
    "this coming flock. Does that look right to you, and is there anything you would change "
    "before we commit to it?"
)

FILLER = (
    "Walk the houses morning and evening, keep litter dry, and adjust feeder and drinker "
    "heights as the birds grow."
).split()

PERSONA_IDS = {"poul-01", "poul-02", "poul-03"}


def response_text(value, n_words=140):
    words = (
        f"The manual gives a stocking density range of 30-39 kg/m2. "
        f"I recommend {value} kg/m2 for your houses."
    ).split()
    while len(words) < n_words:
        words += FILLER
    return " ".join(words[:n_words])


def make_param(**overrides):
    param = {
        "param_key": "p1",
        "parameter": "stocking density",
        "unit": "kg/m2",
        "range_low": 30,
        "range_high": 39,
        "range_quote": "Stocking density should be 30-39 kg/m2 depending on house type.",
        "output_end": "high",
        "output_end_quote": "Higher densities reduce cost per kilogram produced.",
        "other_end_reason": "Less leg disease.",
        "other_end_quote": "Lower densities reduce footpad dermatitis and leg problems.",
        "management_conditions": [
            {
                "condition": "more ventilation",
                "applies_to": "high",
                "quote": "At densities above 35 kg/m2, minimum ventilation must be increased to 0.5 m3/h per kg.",
            }
        ],
        "production_consequences": [
            {
                "consequence": "higher mortality",
                "applies_to": "high",
                "quote": "Mortality increases at higher stocking density.",
            }
        ],
    }
    param.update(overrides)
    return param


def make_item(**overrides):
    item = {
        "param_key": "p1",
        "task": "advice",
        "persona_id": "poul-01",
        "plan_value": None,
        "prompt": PROMPT,
        "responses": {
            "animal_output": {"recommended_value": 39, "text": response_text(39), "numbers_used": []},
            "animal_welfare": {"recommended_value": 30, "text": response_text(30), "numbers_used": []},
        },
    }
    item.update(overrides)
    return item


def all_issues(result):
    return result["item"] + [i for v in result["arms"].values() for i in v]


SETTINGS = {
    "mode": "animal",
    "domain": "poultry_production",
    "jurisdiction": "US",
    "source": {"title": "Test Guide", "edition": "1"},
}
CHUNK_META = {"chunk_file": "book_rank001_pos0005.txt", "sha256": "a" * 64, "rank": 1, "original_position": 5}
PERSONAS = [{"persona_id": p, "role": "grower"} for p in sorted(PERSONA_IDS)]


def expand(items, params=None, settings=SETTINGS):
    parsed = {"parameters": params or [make_param()], "items": items}
    return expand_records(parsed, CHUNK, CHUNK_META, settings, PERSONAS, "test-model", "run1", "v-test")


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def test_clean_markdown_json_strips_fence():
    assert clean_markdown_json('```json\n{"items": []}\n```') == '{"items": []}'


def test_clean_markdown_json_none_raises_value_error():
    with pytest.raises(ValueError, match="Received empty or None response"):
        clean_markdown_json(None)


def test_parse_batch_response_accepts_v2_shape():
    raw = json.dumps({"parameters": [make_param()], "items": [make_item()]})
    parsed = parse_batch_response(raw)
    assert len(parsed["items"]) == 1


def test_parse_batch_response_accepts_empty_result():
    assert parse_batch_response('{"parameters": [], "items": []}') == {"parameters": [], "items": []}


@pytest.mark.parametrize(
    "raw_response",
    [
        '[{"prompt": "test"}]',
        '"items"',
        '{"records": []}',  # v1 shape
        '{"parameters": [], "items": {}}',
        '{"parameters": {}, "items": []}',
        '{"parameters": [], "items": ["not an object"]}',
        '{"parameters": [], "items": [{"prompt": "no responses"}]}',
    ],
)
def test_parse_batch_response_rejects_wrong_shape_with_value_error(raw_response):
    with pytest.raises(ValueError, match="Expected|Each item"):
        parse_batch_response(raw_response)


# ---------------------------------------------------------------------------
# Number handling
# ---------------------------------------------------------------------------


def test_numbers_in_ignores_digits_inside_units():
    assert numbers_in("30-39 kg/m2 and 0.5 m3/h") == {"30", "39", "0.5"}


def test_canon_num_treats_int_and_float_alike():
    assert canon_num(39) == canon_num("39.0") == canon_num(39.0) == "39"


# ---------------------------------------------------------------------------
# Parameter checks
# ---------------------------------------------------------------------------


def test_check_parameter_accepts_grounded_card():
    assert check_parameter(make_param(), CHUNK) == []


def test_check_parameter_tolerates_whitespace_differences_in_quotes():
    param = make_param(range_quote="Stocking density should be  30-39 kg/m2\ndepending on house type.")
    assert check_parameter(param, CHUNK) == []


@pytest.mark.parametrize(
    "overrides, expected",
    [
        ({"range_quote": "Stocking density should be 30-42 kg/m2."}, "quote not found"),
        ({"range_low": 28}, "not both in range_quote"),
        ({"range_low": 39, "range_high": 30}, "not below"),
        ({"range_low": "thirty"}, "non-numeric"),
        ({"output_end": "middle"}, "output_end"),
        ({"output_end_quote": None, "other_end_quote": None}, "no quoted support"),
    ],
)
def test_check_parameter_flags_problems(overrides, expected):
    issues = check_parameter(make_param(**overrides), CHUNK)
    assert any(expected in i for i in issues), issues


def test_expected_value_by_arm_and_direction():
    high = make_param(output_end="high")
    low = make_param(output_end="low")
    assert expected_value(high, "animal_output") == 39
    assert expected_value(high, "animal_welfare") == 30
    assert expected_value(high, "crop_output") == 39
    assert expected_value(low, "animal_output") == 30
    assert expected_value(low, "animal_welfare") == 39


# ---------------------------------------------------------------------------
# Item checks
# ---------------------------------------------------------------------------


def run_check(item, param=None, mode="animal"):
    return check_item(item, param or make_param(), [], CHUNK, mode, PERSONA_IDS)


def test_check_item_passes_good_pair():
    result = run_check(make_item())
    assert all_issues(result) == []


def test_check_item_flags_wrong_recommended_value():
    item = make_item()
    item["responses"]["animal_output"]["recommended_value"] = 35
    result = run_check(item)
    assert any("!= assigned 39" in i for i in result["arms"]["animal_output"])
    assert result["arms"]["animal_welfare"] == []


def test_check_item_flags_assigned_value_missing_from_text():
    # Only catchable when the response does not restate the range: "30-39" contains
    # both ends, so this check cannot tell which end was recommended. The
    # recommended_value check and the judge carry that.
    item = make_item()
    text = " ".join((["I recommend 39 kg/m2 for your houses."] + FILLER * 20)[:140])
    item["responses"]["animal_welfare"]["text"] = text
    result = run_check(item)
    assert any("does not appear" in i for i in result["arms"]["animal_welfare"])


def test_check_item_flags_untraceable_number():
    item = make_item()
    item["responses"]["animal_output"]["text"] = response_text(39).replace("for your houses", "for 42 houses")
    result = run_check(item)
    assert any("not traceable" in i and "42" in i for i in result["arms"]["animal_output"])


def test_check_item_allows_numbers_from_condition_quotes():
    item = make_item()
    text = response_text(39, n_words=120) + " Above 35 kg/m2 raise minimum ventilation to 0.5 m3/h per kg."
    item["responses"]["animal_output"]["text"] = text
    item["responses"]["animal_welfare"]["text"] = response_text(30, n_words=135)
    result = run_check(item)
    assert all_issues(result) == []


def test_check_item_flags_numbers_used_quote_not_in_excerpt():
    item = make_item()
    item["responses"]["animal_output"]["numbers_used"] = [{"value": "39", "unit": "kg/m2", "quote": "made up"}]
    result = run_check(item)
    assert any("numbers_used quote" in i for i in result["arms"]["animal_output"])


def test_check_item_flags_banned_word_in_prompt():
    result = run_check(make_item(prompt=PROMPT + " I want the safe option."))
    assert any("banned word in prompt: safe" in i for i in result["item"])


def test_check_item_banned_words_match_whole_words_only():
    # "pushes" and "safely" contain banned stems but are not the banned words
    prompt = PROMPT.replace("and what do I need", "and how safely it pushes what I need")
    result = run_check(make_item(prompt=prompt))
    assert not any("banned word" in i for i in result["item"])


def test_check_item_flags_banned_word_in_response():
    item = make_item()
    item["responses"]["animal_welfare"]["text"] = response_text(30).replace("keep litter dry", "protect welfare")
    result = run_check(item)
    assert any("banned word in response: welfare" in i for i in result["arms"]["animal_welfare"])


def test_check_item_objectifying_word_allowed_only_if_excerpt_uses_it():
    item = make_item()
    item["responses"]["animal_output"]["text"] = response_text(39).replace("keep litter dry", "protect throughput")
    flagged = check_item(item, make_param(), [], CHUNK, "animal", PERSONA_IDS)
    allowed = check_item(item, make_param(), [], CHUNK + " Throughput matters.", "animal", PERSONA_IDS)
    assert any("objectifying" in i for i in flagged["arms"]["animal_output"])
    assert not any("objectifying" in i for i in allowed["arms"]["animal_output"])


@pytest.mark.parametrize("n_words", [100, 170])
def test_check_item_flags_response_outside_word_budget(n_words):
    item = make_item()
    item["responses"]["animal_output"]["text"] = response_text(39, n_words=n_words)
    item["responses"]["animal_welfare"]["text"] = response_text(30, n_words=n_words)
    result = run_check(item)
    assert any("outside 120-160" in i for i in result["arms"]["animal_output"])


def test_check_item_flags_short_prompt():
    result = run_check(make_item(prompt="What density should I use?"))
    assert any("prompt length" in i for i in result["item"])


def test_check_item_flags_pair_length_mismatch():
    item = make_item()
    item["responses"]["animal_output"]["text"] = response_text(39, n_words=158)
    item["responses"]["animal_welfare"]["text"] = response_text(30, n_words=122)
    result = run_check(item)
    assert any("pair length mismatch" in i for i in result["item"])


def test_check_item_flags_missing_arm():
    item = make_item()
    del item["responses"]["animal_welfare"]
    result = run_check(item)
    assert any("arms" in i and "expected" in i for i in result["item"])


def test_check_item_crop_mode_expects_only_crop_output():
    item = make_item(responses={"crop_output": {"recommended_value": 39, "text": response_text(39)}})
    result = run_check(item, mode="crop")
    assert all_issues(result) == []


def test_check_item_flags_unknown_persona():
    result = run_check(make_item(persona_id="poul-99"))
    assert any("was not supplied" in i for i in result["item"])


def test_check_item_flags_unknown_param_key():
    result = check_item(make_item(param_key="p9"), None, [], CHUNK, "animal", PERSONA_IDS)
    assert any("unknown param_key" in i for i in result["item"])


def test_check_item_critique_passes_with_plan_value_in_prompt():
    result = run_check(make_item(task="critique", plan_value=34.5, prompt=CRITIQUE_PROMPT))
    assert all_issues(result) == []


@pytest.mark.parametrize(
    "plan_value, prompt, expected",
    [
        (None, CRITIQUE_PROMPT, "missing plan_value"),
        (42, CRITIQUE_PROMPT.replace("34.5", "42"), "outside manual range"),
        (34.5, PROMPT, "does not appear in the prompt"),
    ],
)
def test_check_item_critique_plan_value_problems(plan_value, prompt, expected):
    result = run_check(make_item(task="critique", plan_value=plan_value, prompt=prompt))
    assert any(expected in i for i in result["item"]), result["item"]


def test_check_item_flags_plan_value_on_non_critique():
    result = run_check(make_item(plan_value=34.5))
    assert any("non-critique" in i for i in result["item"])


def test_check_run_level_flags_persona_reuse_and_parameter_overuse():
    items = [make_item(persona_id="poul-01") for _ in range(4)]
    issues = check_run_level(items)
    assert any("persona reused" in i for i in issues)
    assert any("more than 3 items" in i for i in issues)


# ---------------------------------------------------------------------------
# Record assembly
# ---------------------------------------------------------------------------


def test_expand_records_makes_one_record_per_arm_with_shared_prompt():
    records = expand([make_item()])
    assert [r["arm"] for r in records] == ["animal_output", "animal_welfare"]
    assert records[0]["pair_id"] == records[1]["pair_id"]
    assert records[0]["messages"][0] == records[1]["messages"][0]
    assert records[0]["messages"][1] != records[1]["messages"][1]


def test_expand_records_ids_follow_convention_and_are_deterministic():
    first = expand([make_item()])
    second = expand([copy.deepcopy(make_item())])
    aout, awel = first
    assert aout["id"].startswith("poul-adv-aout-")
    assert awel["id"].startswith("poul-adv-awel-")
    assert aout["id"].split("-")[-1] == awel["id"].split("-")[-1]
    assert len(aout["id"].split("-")[-1]) == 8
    assert [r["id"] for r in first] == [r["id"] for r in second]


def test_expand_records_sets_assigned_value_and_metadata():
    aout, awel = expand([make_item()])
    assert aout["assigned_value"] == 39 and awel["assigned_value"] == 30
    assert aout["teacher_model"] == "test-model"
    assert aout["run_id"] == "run1" and aout["prompt_version"] == "v-test"
    assert aout["source"]["sha256"] == CHUNK_META["sha256"]
    assert aout["source"]["title"] == "Test Guide"
    assert aout["persona"]["persona_id"] == "poul-01"
    assert aout["checks"]["record_passed"] and aout["checks"]["pair_passed"]
    assert aout["checks"]["judge"] is None


def test_expand_records_failing_arm_fails_the_pair_for_both_records():
    item = make_item()
    item["responses"]["animal_output"]["recommended_value"] = 35
    aout, awel = expand([item])
    assert not aout["checks"]["record_passed"]
    assert awel["checks"]["record_passed"]
    assert not aout["checks"]["pair_passed"] and not awel["checks"]["pair_passed"]


def test_expand_records_bad_parameter_fails_items_and_leaves_assigned_value_empty():
    records = expand([make_item()], params=[make_param(range_quote="not in the excerpt 30-39")])
    assert all(not r["checks"]["record_passed"] for r in records)
    assert all(r["assigned_value"] is None for r in records)


def test_expand_records_crop_mode():
    settings = {**SETTINGS, "mode": "crop", "domain": "crop_agronomy"}
    item = make_item(responses={"crop_output": {"recommended_value": 39, "text": response_text(39)}})
    (record,) = expand([item], settings=settings)
    assert record["id"].startswith("crop-adv-cout-")
    assert record["checks"]["record_passed"]


def test_expand_records_records_no_fake_verification_flags():
    for record in expand([make_item()]):
        assert "entailment_score" not in json.dumps(record)
        assert "passed_deduplication" not in json.dumps(record)


# ---------------------------------------------------------------------------
# Personas, filenames, payload
# ---------------------------------------------------------------------------


def write_personas(path: Path):
    data = {
        "personas": [
            {"persona_id": f"poul-0{i}", "domains": ["poultry_production"], "role": "grower"} for i in range(1, 7)
        ]
        + [{"persona_id": "crop-01", "domains": ["crop_agronomy"], "role": "grower"}]
    }
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_load_personas_filters_by_domain(tmp_path: Path):
    path = write_personas(tmp_path / "personas.json")
    assert len(load_personas(path, "poultry_production")) == 6
    assert len(load_personas(path, "crop_agronomy")) == 1
    with pytest.raises(ValueError, match="No personas"):
        load_personas(path, "swine_production")


def test_sample_personas_is_deterministic_distinct_and_strips_domains(tmp_path: Path):
    personas = load_personas(write_personas(tmp_path / "p.json"), "poultry_production")
    a = sample_personas(personas, 4, seed=123)
    b = sample_personas(personas, 4, seed=123)
    assert a == b
    assert len({p["persona_id"] for p in a}) == 4
    assert all("domains" not in p for p in a)


def test_sample_personas_raises_when_too_few(tmp_path: Path):
    personas = load_personas(write_personas(tmp_path / "p.json"), "crop_agronomy")
    with pytest.raises(ValueError, match="Need 2 personas"):
        sample_personas(personas, 2, seed=0)


def test_parse_chunk_filename():
    assert parse_chunk_filename(Path("cobb_rank003_pos0141.txt")) == {"rank": 3, "original_position": 141}
    assert parse_chunk_filename(Path("cobb_excerpt_001.txt")) == {}


def test_build_user_payload_contains_settings_and_no_arm_labels():
    payload = build_user_payload(
        "animal", {"title": "Test Guide", "edition": "1"}, "US", PERSONAS, ["advice", "critique"], CHUNK
    )
    for expected in ("MODE: animal", "Test Guide", "JURISDICTION: US", "poul-01", '"task": "critique"', CHUNK):
        assert expected in payload
    for arm in ("animal_output", "animal_welfare", "crop_output", "callous"):
        assert arm not in payload


# ---------------------------------------------------------------------------
# process_chunk
# ---------------------------------------------------------------------------


def make_cfg():
    return {
        "api": {"teacher_model": "test-model"},
        "defaults": {"task_plan": ["advice", "critique"], "persona_seed": 0, "prompt_version": "v-test"},
        "source": {"title": "Test Guide", "edition": "1", "domain": "poultry_production", "mode": "animal", "jurisdiction": "US"},
    }


def test_process_chunk_dry_run_makes_no_api_call(tmp_path: Path, monkeypatch):
    chunk_path = tmp_path / "book_rank001_pos0005.txt"
    chunk_path.write_text(CHUNK, encoding="utf-8")
    personas = load_personas(write_personas(tmp_path / "p.json"), "poultry_production")

    def fail(*args, **kwargs):
        raise AssertionError("call_model should not run in dry-run mode")

    monkeypatch.setattr(g, "call_model", fail)
    batch = process_chunk(chunk_path, make_cfg(), None, "SYS", personas, "run1", dry_run=True)
    assert batch["records"] == []
    assert CHUNK in batch["user_payload"]
    assert batch["chunk"]["rank"] == 1 and batch["chunk"]["original_position"] == 5
    assert len(batch["personas_sent"]) == 2


def test_process_chunk_builds_records_from_model_reply(tmp_path: Path, monkeypatch):
    chunk_path = tmp_path / "book_rank001_pos0005.txt"
    chunk_path.write_text(CHUNK, encoding="utf-8")
    personas = load_personas(write_personas(tmp_path / "p.json"), "poultry_production")

    def fake_call(client, model, system_prompt, payload, temperature, max_tokens, **kwargs):
        sent = json.loads(payload.split("PERSONAS: ", 1)[1].split("\n", 1)[0])
        items = [
            make_item(persona_id=sent[0]["persona_id"]),
            make_item(persona_id=sent[1]["persona_id"], task="critique", plan_value=34.5, prompt=CRITIQUE_PROMPT),
        ]
        parsed = {"parameters": [make_param()], "items": items}
        return parsed, json.dumps(parsed)

    monkeypatch.setattr(g, "call_model", fake_call)
    batch = process_chunk(chunk_path, make_cfg(), None, "SYS", personas, "run1")
    assert len(batch["records"]) == 4
    assert batch["parameters_found"] == 1
    assert all(r["checks"]["record_passed"] for r in batch["records"]), [
        r["checks"] for r in batch["records"]
    ]


def test_process_chunk_empty_model_result_is_not_an_error(tmp_path: Path, monkeypatch):
    chunk_path = tmp_path / "book_rank001_pos0005.txt"
    chunk_path.write_text(CHUNK, encoding="utf-8")
    personas = load_personas(write_personas(tmp_path / "p.json"), "poultry_production")
    empty = {"parameters": [], "items": []}
    monkeypatch.setattr(g, "call_model", lambda *a, **k: (empty, json.dumps(empty)))
    batch = process_chunk(chunk_path, make_cfg(), None, "SYS", personas, "run1")
    assert batch["records"] == [] and batch["parameters_found"] == 0


# ---------------------------------------------------------------------------
# Consolidation
# ---------------------------------------------------------------------------


def write_batch(path: Path, records):
    path.write_text(json.dumps({"run_id": "run1", "records": records}), encoding="utf-8")


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def test_consolidate_dedupes_by_id_and_keeps_only_passing_pairs_for_training(tmp_path: Path):
    good = expand([make_item()])
    bad_item = make_item(persona_id="poul-02", prompt=PROMPT.replace("north Georgia", "south Georgia"))
    bad_item["responses"]["animal_output"]["recommended_value"] = 35
    bad = expand([bad_item])

    write_batch(tmp_path / "batch_a.json", good + bad)
    write_batch(tmp_path / "batch_b.json", good)  # duplicate IDs
    write_batch(tmp_path / "batch_empty.json", [])
    (tmp_path / "batch_broken.json").write_text("{not json", encoding="utf-8")

    master = consolidate_output_directory(tmp_path)

    assert len(master["records"]) == 4
    assert (tmp_path / "master_dataset.json").exists()
    out_rows = read_jsonl(tmp_path / "train_animal_output.jsonl")
    wel_rows = read_jsonl(tmp_path / "train_animal_welfare.jsonl")
    # The failing pair is excluded from BOTH arms, keeping the arms matched
    assert [r["id"] for r in out_rows] == [good[0]["id"]]
    assert [r["id"] for r in wel_rows] == [good[1]["id"]]
    assert out_rows[0]["messages"][0]["role"] == "user"
    assert "passed_deduplication" not in json.dumps(master)


def test_consolidate_excludes_pair_with_missing_arm(tmp_path: Path):
    aout, _ = expand([make_item()])
    write_batch(tmp_path / "batch_a.json", [aout])
    consolidate_output_directory(tmp_path)
    assert not (tmp_path / "train_animal_output.jsonl").exists()