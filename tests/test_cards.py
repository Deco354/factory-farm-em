"""Tests for the cards pipeline (cards.py and generate_cards.py)."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from fragile_compassion.sft_document_generation import cards as cd
from fragile_compassion.sft_document_generation import generate as g
from fragile_compassion.sft_document_generation import generate_cards as gc

BOOK = (
    "Any time stocking densities increase above 28 kg/m2 heat trapped below the birds can build up very quickly. "
    "Increased bird temperatures will result in higher levels of panting, reduce feed intake and subsequent "
    "decreases in daily gains. Do not use stocking densities higher than 42 kg/m2 because flock heat removal "
    "becomes very difficult. Stocking densities above 28 kg/m2 require the stockman to constantly monitor bird "
    "behavior. National requirements must always be considered.\n"
    "Optimum recommended time for feed withdrawal is 8 to 12 hours. Less than 8 hours is a waste of the "
    "undigested feed. Shorter withdrawal reduces time without feed."
)


def make_card(**overrides):
    card = {
        "card_id": "test-density",
        "parameter": "final stocking density",
        "parameter_category": "space_allowance",
        "unit": "kg/m2",
        "range_low": 28,
        "range_high": 42,
        "range_quote": [
            "Any time stocking densities increase above 28 kg/m2",
            "Do not use stocking densities higher than 42 kg/m2",
        ],
        "output_end": "high",
        "output_reason_basis": "unit",
        "output_end_reason": None,
        "output_end_quote": None,
        "other_reason_basis": "quote",
        "other_end_reason": "Above 28 kg/m2, heat trapped below the birds can build up very quickly.",
        "other_end_quote": "heat trapped below the birds can build up very quickly.",
        "management_conditions": [
            {
                "condition": "Monitor bird behavior above 28 kg/m2",
                "applies_to": "high",
                "quote": "require the stockman to constantly monitor bird behavior.",
            },
            {
                "condition": "Consider national requirements",
                "applies_to": "both",
                "quote": "National requirements must always be considered.",
            },
        ],
        "production_consequences": [
            {
                "consequence": "More panting and lower daily gains",
                "applies_to": "high",
                "quote": "Increased bird temperatures will result in higher levels of panting, reduce feed intake and subsequent decreases in daily gains.",
            },
        ],
        "page": 40,
        "reviewed_by": "CB",
    }
    card.update(overrides)
    return card


SRC = {
    "title": "Test Guide",
    "edition": "1",
    "domain": "poultry_production",
    "mode": "animal",
    "jurisdiction": "US",
    "book_text": "book.txt",
}

# ---------------------------------------------------------------------------
# Card checks
# ---------------------------------------------------------------------------


def test_good_card_passes():
    assert cd.check_card(make_card(), BOOK) == []


def test_single_string_range_quote_passes():
    card = make_card(
        card_id="test-withdrawal",
        parameter="feed withdrawal",
        parameter_category="feeding",
        unit="hours",
        range_low=8,
        range_high=12,
        range_quote="Optimum recommended time for feed withdrawal is 8 to 12 hours",
        output_reason_basis="quote",
        output_end_reason="Less than 8 hours wastes undigested feed.",
        output_end_quote="Less than 8 hours is a waste of the undigested feed.",
        other_end_reason="Shorter withdrawal reduces time without feed.",
        other_end_quote="Shorter withdrawal reduces time without feed.",
        management_conditions=[],
        production_consequences=[],
    )
    assert cd.check_card(card, BOOK) == []


@pytest.mark.parametrize(
    "overrides, expected",
    [
        ({"range_quote": "Do not use stocking densities higher than 45 kg/m2"}, "quote not found"),
        ({"other_end_quote": "made-up quote"}, "quote not found"),
        ({"range_low": 30}, "not both in range_quote"),
        ({"parameter_category": "environment"}, "'unit' not allowed for category"),
        ({"output_reason_basis": "quote"}, "output_end_quote is empty"),
        ({"output_reason_basis": "quote"}, "output_end_reason is empty"),
        ({"output_end_reason": "More kilos per square metre."}, "must be empty"),
        (
            {"other_end_reason": "Below 25 kg/m2 heat stays low."},
            "numbers not in the card's quotes",
        ),
        ({"other_reason_basis": "guess"}, "must be 'quote' or 'unit'"),
        ({"other_reason_basis": "quote", "other_end_quote": None}, "other_end_quote is empty"),
        ({"parameter_category": "care_effort"}, "excluded"),
        ({"unit": ""}, "missing field: unit"),
    ],
)
def test_card_problems_are_reported(overrides, expected):
    issues = cd.check_card(make_card(**overrides), BOOK)
    assert any(expected in i for i in issues), issues


def test_resolved_reason_uses_fixed_sentence_for_unit_basis():
    card = make_card()
    assert (
        cd.resolved_reason(card, "output") == "At 42 kg/m2, the same floor area carries more stock."
    )
    assert cd.resolved_reason(card, "other") == card["other_end_reason"]
    both_unit = make_card(other_reason_basis="unit", other_end_reason=None)
    assert cd.resolved_reason(both_unit, "other") == "At 28 kg/m2, the stock has more floor area."


def test_resolved_reason_follows_output_end_direction():
    card = make_card(
        parameter_category="animals_per_equipment",
        unit="birds per nipple",
        range_low=20,
        range_high=25,
        output_end="high",
    )
    assert (
        cd.resolved_reason(card, "output")
        == "At 25 birds per nipple, each piece of equipment serves more animals."
    )


def test_is_reviewed():
    assert cd.is_reviewed(make_card())
    assert not cd.is_reviewed(make_card(reviewed_by=None))
    assert not cd.is_reviewed(make_card(reviewed_by="  "))


def test_select_cards_skips_unreviewed_unless_allowed():
    cards = [
        make_card(card_id="a"),
        make_card(card_id="b", reviewed_by=None),
        make_card(card_id="c"),
    ]
    selected, skipped = gc.select_cards(cards, None, None, allow_unreviewed=False)
    assert [c["card_id"] for c in selected] == ["a", "c"] and skipped == ["b"]
    selected, skipped = gc.select_cards(cards, ["b"], None, allow_unreviewed=True)
    assert [c["card_id"] for c in selected] == ["b"] and skipped == []
    selected, _ = gc.select_cards(cards, None, 1, allow_unreviewed=False)
    assert [c["card_id"] for c in selected] == ["a"]


def test_bad_applies_to_is_reported():
    card = make_card()
    card["management_conditions"][0]["applies_to"] = "middle"
    assert any("applies_to" in i for i in cd.check_card(card, BOOK))


def write_card_file(tmp_path: Path, cards, source=None) -> Path:
    (tmp_path / "book.txt").write_text(BOOK, encoding="utf-8")
    path = tmp_path / "cards.yaml"
    path.write_text(
        yaml.safe_dump({"source": source or SRC, "cards": cards}, allow_unicode=True),
        encoding="utf-8",
    )
    return path


def test_check_card_file_reports_per_card_and_duplicates(tmp_path: Path):
    path = write_card_file(tmp_path, [make_card(), make_card(range_low=30)])
    report = cd.check_card_file(path, tmp_path)
    assert any("duplicate card_id" in i for i in report["<file>"])


def test_check_card_file_missing_book(tmp_path: Path):
    path = write_card_file(tmp_path, [make_card()], source={**SRC, "book_text": "nope.txt"})
    report = cd.check_card_file(path, tmp_path)
    assert any("book_text not found" in i for i in report["<file>"])


# ---------------------------------------------------------------------------
# What each arm sees
# ---------------------------------------------------------------------------


def test_end_of_by_arm():
    card = make_card()
    assert cd.end_of(card, "animal_output") == "high"
    assert cd.end_of(card, "animal_control") == "low"
    assert cd.end_of(card, "crop_control") == "high"


def test_response_inputs_only_include_own_end():
    card = make_card()
    out = gc.response_inputs(card, "animal_output")
    ctl = gc.response_inputs(card, "animal_control")
    assert out["value"] == 42 and ctl["value"] == 28
    assert (
        len(out["conditions"]) == 2 and len(ctl["conditions"]) == 1
    )  # 'high' condition only for output arm
    assert out["consequences"] and not ctl["consequences"]
    assert card["other_end_quote"] in ctl["quotes"] and card["other_end_quote"] not in out["quotes"]
    assert out["reason"] == "At 42 kg/m2, the same floor area carries more stock."
    assert ctl["reason"] == card["other_end_reason"]


def test_response_payload_mentions_only_own_value():
    card = make_card()
    payload = gc.build_response_payload(card, "animal_control", "How dense?", "advice", None)
    assert "RECOMMENDED_VALUE: 28 kg/m2" in payload
    assert "REASON_FOCUS: animals" in payload
    assert cd.resolved_reason(card, "output") not in payload


def test_prompt_payload_hides_direction():
    card = make_card()
    payload = gc.build_prompt_payload(
        card, "broiler chickens", [{"item_index": 0, "task": "advice"}]
    )
    for hidden in (
        cd.resolved_reason(card, "output"),
        card["other_end_reason"],
        "output_end",
        "42",
        "28",
    ):
        assert hidden not in payload


# ---------------------------------------------------------------------------
# Plan values and recommendation parsing
# ---------------------------------------------------------------------------


def test_plan_values_rotate_low_high_mid_for_critiques():
    vals = gc.plan_values(make_card(), ["advice", "critique", "critique", "tutoring", "critique"])
    assert vals == [None, 28, 42, None, 35]


def test_plan_values_midpoint_rounds_down_for_integer_ranges():
    vals = gc.plan_values(make_card(range_low=20, range_high=25), ["critique"] * 3)
    assert vals == [20, 25, 22]


@pytest.mark.parametrize(
    "text, expected",
    [
        ("The guide gives 28 to 42 kg/m2. I recommend 42 kg/m2 for your houses.", 42),
        ("The guide gives 28 to 42 kg/m2. I recommend 28 kg/m2 for your houses.", 28),
        ("I recommend a density between 28 and 42 kg/m2.", None),  # names both ends
        ("Use 42 kg/m2.", None),  # no recommendation sentence
        ("For your plan of 35 kg/m2, I recommend moving to 42 kg/m2.", 42),
    ],
)
def test_recommended_value_from_text(text, expected):
    assert gc.recommended_value_from_text(text, 42, 28) == expected


# ---------------------------------------------------------------------------
# process_card with a fake model
# ---------------------------------------------------------------------------

FILLER = (
    "Keep litter dry, walk the houses morning and evening, and adjust feeder and drinker heights "
    "as the birds grow."
).split()


def reply_text(value: int, task: str, extra: str = "") -> str:
    words = (
        f"The guide gives a range of 28 to 42 kg/m2. I recommend {value} kg/m2 for your houses. "
        + extra
    ).split()
    target = 130 if task == "tutoring" else 110
    while len(words) < target:
        words += FILLER
    return " ".join(words[:target])


PROMPT_TEXT = (
    "We run four broiler houses in north Georgia on contract and are planning the next placement. "
    "What final stocking density should we plan for, and what should we watch for along the way?"
)


def critique_prompt(value) -> str:
    return (
        f"We are planning to place birds at {g.canon_num(value)} kg/m2 in our tunnel houses this coming flock. "
        "Does that look right to you, and is there anything you would change before we commit to it?"
    )


def fake_call_factory(fail_arm=None):
    state = {"items": None}

    def fake_call(client, model, system, payload, temperature, max_tokens, reasoning=None):
        if payload.startswith("PARAMETER:"):
            items = json.loads(payload.split("ITEMS: ", 1)[1])
            state["items"] = items
            prompts = [
                {
                    "item_index": it["item_index"],
                    "prompt": critique_prompt(it["plan_value"])
                    if it["task"] == "critique"
                    else PROMPT_TEXT,
                }
                for it in items
            ]
            return {"prompts": prompts}, json.dumps({"prompts": prompts})
        value = int(float(payload.split("RECOMMENDED_VALUE: ", 1)[1].split()[0]))
        task = payload.split("TASK: ", 1)[1].split("\n", 1)[0]
        focus = payload.split("REASON_FOCUS: ", 1)[1].split("\n", 1)[0]
        if fail_arm and focus == fail_arm:
            raise g.EmptyReplyError("Empty reply from model (finish_reason=stop).")
        return {"text": reply_text(value, task)}, "{}"

    return fake_call


SETTINGS = {
    "task_plan": ["advice", "critique"],
    "persona_seed": 0,
    "model": "test-model",
    "temperature": 0.3,
    "max_tokens": 1000,
    "reasoning": {"effort": "low"},
    "prompt_version": "cards-test",
    "prompt_writer": "PW",
    "response_writer": "RW",
}
PERSONAS = [{"persona_id": f"poul-0{i}", "role": "grower"} for i in range(1, 5)]


def test_process_card_builds_matched_passing_pairs():
    batch = gc.process_card(
        make_card(), SRC, BOOK, SETTINGS, None, PERSONAS, "run1", call=fake_call_factory()
    )
    records = batch["records"]
    assert len(records) == 4 and batch["declined"] == []
    assert all(r["checks"]["record_passed"] for r in records), [r["checks"] for r in records]
    by_pair = {}
    for r in records:
        by_pair.setdefault(r["pair_id"], []).append(r)
    for pair in by_pair.values():
        assert {r["arm"] for r in pair} == {"animal_output", "animal_control"}
        assert pair[0]["messages"][0] == pair[1]["messages"][0]
    out = next(r for r in records if r["arm"] == "animal_output")
    assert out["assigned_value"] == 42 and out["recommended_value"] == 42
    assert out["id"].startswith("poul-adv-aout-") or out["id"].startswith("poul-cri-aout-")
    assert out["source"]["card_id"] == "test-density" and out["pipeline"] == "cards"


def test_process_card_records_declined_arm_and_fails_pair():
    batch = gc.process_card(
        make_card(),
        SRC,
        BOOK,
        SETTINGS,
        None,
        PERSONAS,
        "run1",
        call=fake_call_factory(fail_arm="animals"),
    )
    assert {d["arm"] for d in batch["declined"]} == {"animal_control"}
    assert all(r["arm"] == "animal_output" for r in batch["records"])
    assert not any(r["checks"]["pair_passed"] for r in batch["records"])


def test_process_card_rejects_bad_card_without_calling_model():
    def boom(*a, **k):
        raise AssertionError("should not call the model")

    batch = gc.process_card(
        make_card(range_low=30), SRC, BOOK, SETTINGS, None, PERSONAS, "run1", call=boom
    )
    assert batch["card_issues"] and batch["records"] == []


def test_process_card_dry_run():
    batch = gc.process_card(make_card(), SRC, BOOK, SETTINGS, None, PERSONAS, "run1", dry_run=True)
    assert "prompt_payload" in batch and set(batch["response_payload_examples"]) == {
        "animal_output",
        "animal_control",
    }


def test_word_budget_replaces_generate_budget():
    issues = gc.apply_card_word_budget(
        ["response length 100 words outside 120-160", "other"], "w " * 100, "advice"
    )
    assert issues == ["other"]
    assert gc.apply_card_word_budget([], "w " * 60, "advice") == [
        "response length 60 words outside 90-150"
    ]


# ---------------------------------------------------------------------------
# call_json
# ---------------------------------------------------------------------------


class FakeClient:
    def __init__(self, content, finish="stop"):
        self.kwargs = None
        self.calls = 0
        self._resp = SimpleNamespace(
            choices=[
                SimpleNamespace(finish_reason=finish, message=SimpleNamespace(content=content))
            ],
            usage=None,
        )
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.kwargs, self.calls = kwargs, self.calls + 1
        return self._resp


def test_call_json_passes_reasoning_cap(monkeypatch):
    client = FakeClient('{"text": "hi"}')
    parsed, _ = gc.call_json(client, "m", "s", "u", 0.3, 100, reasoning={"effort": "low"})
    assert parsed == {"text": "hi"}
    assert client.kwargs["extra_body"] == {"reasoning": {"effort": "low"}}


def test_call_json_does_not_retry_truncation(monkeypatch):
    monkeypatch.setattr(gc.time, "sleep", lambda s: None)
    client = FakeClient('{"te', finish="length")
    with pytest.raises(g.TruncatedReplyError):
        gc.call_json(client, "m", "s", "u", 0.3, 100)
    assert client.calls == 1


def test_call_json_rejects_non_object(monkeypatch):
    monkeypatch.setattr(gc.time, "sleep", lambda s: None)
    client = FakeClient("[1, 2]")
    with pytest.raises(ValueError, match="JSON object"):
        gc.call_json(client, "m", "s", "u", 0.3, 100, max_retries=2)
    assert client.calls == 2


# ---------------------------------------------------------------------------
# Consolidation reuses generate.py
# ---------------------------------------------------------------------------


def test_cards_batches_consolidate_into_training_files(tmp_path: Path):
    batch = gc.process_card(
        make_card(), SRC, BOOK, SETTINGS, None, PERSONAS, "run1", call=fake_call_factory()
    )
    (tmp_path / "batch_test-density.json").write_text(json.dumps(batch), encoding="utf-8")
    g.consolidate_output_directory(tmp_path)
    out = (tmp_path / "train_animal_output.jsonl").read_text(encoding="utf-8").splitlines()
    ctl = (tmp_path / "train_animal_control.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(out) == len(ctl) == 2
