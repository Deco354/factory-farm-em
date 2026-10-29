"""Tests for choice cards: two named practices instead of a numeric range."""

from fragile_compassion.sft_document_generation import cards as cd
from fragile_compassion.sft_document_generation import draft_cards as dc
from fragile_compassion.sft_document_generation import generate as g
from fragile_compassion.sft_document_generation import generate_cards as gc

BOOK = (
    "Finishing pigs can be fed ad libitum or restricted. Restricted feeding improves feed "
    "conversion and carcass leanness. With ad libitum feeding the pigs eat to appetite and "
    "grow faster. Provide enough feeder space for every pig to eat at the same time."
)


def make_choice(**overrides):
    card = {
        "card_id": "test-feeding-method",
        "parameter": "feeding method for finishing pigs",
        "parameter_category": "choice",
        "option_low": "ad libitum",
        "option_high": "restricted",
        "range_quote": "Finishing pigs can be fed ad libitum or restricted.",
        "output_end": "high",
        "output_reason_basis": "quote",
        "output_end_reason": "Restricted feeding improves feed conversion and carcass leanness.",
        "output_end_quote": "Restricted feeding improves feed conversion and carcass leanness.",
        "other_reason_basis": "quote",
        "other_end_reason": "With ad libitum feeding the pigs eat to appetite.",
        "other_end_quote": "With ad libitum feeding the pigs eat to appetite and grow faster.",
        "management_conditions": [
            {
                "condition": "Provide enough feeder space for every pig to eat at once",
                "applies_to": "high",
                "quote": "Provide enough feeder space for every pig to eat at the same time.",
            }
        ],
        "production_consequences": [],
        "reviewed_by": "CB",
    }
    card.update(overrides)
    return card


SRC = {
    "title": "Test Swine Guide",
    "edition": "1",
    "domain": "swine_production",
    "mode": "animal",
    "jurisdiction": "US",
    "book_text": "book.txt",
}
SETTINGS = {
    "task_plan": ["advice", "critique", "critique", "tutoring"],
    "persona_seed": 0,
    "model": "test-model",
    "temperature": 0.3,
    "max_tokens": 1000,
    "reasoning": {"effort": "low"},
    "prompt_version": "cards-test",
    "prompt_writer": "PW",
    "response_writer": "RW",
}
PERSONAS = [{"persona_id": f"swin-0{i}", "role": "grower"} for i in range(1, 5)]
FILLER = " Keep records of intake and adjust the feeders as the pigs grow." * 12


def test_good_choice_card_passes():
    assert cd.check_card(make_choice(), BOOK) == []


def test_choice_card_needs_both_options():
    issues = cd.check_card(make_choice(option_low=None), BOOK)
    assert "missing field: option_low" in issues


def test_choice_card_does_not_need_unit_or_range():
    card = make_choice()
    assert "unit" not in card and "range_low" not in card
    assert cd.check_card(card, BOOK) == []


def test_option_must_be_worded_as_in_the_quotes():
    issues = cd.check_card(make_choice(option_high="limited feeding"), BOOK)
    assert any("option_high 'limited feeding' is not worded" in i for i in issues)


def test_options_must_be_distinguishable():
    issues = cd.check_card(make_choice(option_high="ad libitum feeding"), BOOK)
    assert any("neither may contain the other" in i for i in issues)


def test_choice_reasons_must_be_quoted():
    issues = cd.check_card(make_choice(output_reason_basis="unit", output_end_reason=None), BOOK)
    assert any("'unit' not allowed for category 'choice'" in i for i in issues)


def test_choice_end_values_and_arms():
    card = make_choice()
    assert cd.end_value(card, "low") == "ad libitum"
    assert gc.response_inputs(card, "animal_output")["value"] == "restricted"
    assert gc.response_inputs(card, "animal_control")["value"] == "ad libitum"
    assert g.expected_value(card, "animal_control") == "ad libitum"


def test_choice_plan_values_alternate_options():
    vals = gc.plan_values(make_choice(), ["advice", "critique", "critique", "critique"])
    assert vals == [None, "ad libitum", "restricted", "ad libitum"]


def test_choice_prompt_payload_lists_options_without_direction():
    card = make_choice()
    payload = gc.build_prompt_payload(card, "pigs", [{"item_index": 0, "task": "advice"}])
    assert 'OPTIONS: ["ad libitum", "restricted"]' in payload
    for hidden in (card["output_end_reason"], card["other_end_reason"], "output_end", "None"):
        assert hidden not in payload


def test_choice_response_payload():
    card = make_choice()
    payload = gc.build_response_payload(card, "animal_output", "Which?", "critique", "ad libitum")
    assert "RECOMMENDED_OPTION: restricted" in payload
    assert "CLIENT PLAN: ad libitum" in payload
    assert "RANGE:" not in payload and "RECOMMENDED_VALUE" not in payload
    assert card["other_end_reason"] not in payload


def test_recommended_option_from_text():
    read = gc.recommended_option_from_text
    assert (
        read("The manual allows both. I recommend restricted feeding.", "restricted", "ad libitum")
        == "restricted"
    )
    assert (
        read("I recommend ad libitum feeding, not restricted.", "restricted", "ad libitum")
        == "ad libitum"
    )
    assert (
        read(
            "I recommend moving from ad libitum to restricted feeding.", "restricted", "ad libitum"
        )
        == "restricted"
    )
    assert read("Feed them well.", "restricted", "ad libitum") is None


def reply(option: str, task: str) -> str:
    opening = "Your plan works. " if task == "critique" else ""
    return f"{opening}The manual allows either method. I recommend {option} feeding for your pigs.{FILLER}"


def records_for(card, wrong_arm=None):
    prepared = gc.prepare_card(card, SRC, BOOK, SETTINGS, PERSONAS)
    assert not prepared["card_issues"], prepared["card_issues"]
    ids = {p["persona_id"] for p in prepared["personas"]}
    records = []
    for item in prepared["items"]:
        prompt = (
            "We run two finishing barns in northwest Iowa on contract and are setting up "
            "feeding for the next group of pigs that arrives in three weeks."
        )
        if item["task"] == "critique":
            prompt += f" Right now we feed {item['plan_value']}. What do you think of that?"
        responses = {}
        for arm in ("animal_output", "animal_control"):
            value = gc.response_inputs(card, arm)["value"]
            if arm == wrong_arm:
                value = gc.response_inputs(
                    card, "animal_control" if arm == "animal_output" else "animal_output"
                )["value"]
            words = reply(value, item["task"]).split()
            text = " ".join(words[: 120 if item["task"] != "tutoring" else 150])
            responses[arm] = {"text": text, "retried": False}
        records += gc.build_item_records(card, SRC, SETTINGS, item, prompt, responses, ids, "run1")
    return records


def test_choice_records_pass_and_store_option_wording():
    records = records_for(make_choice())
    assert records and all(r["checks"]["record_passed"] for r in records), [
        r["checks"] for r in records
    ]
    out = next(r for r in records if r["arm"] == "animal_output")
    assert out["assigned_value"] == "restricted" and out["recommended_value"] == "restricted"
    crit = [r for r in records if r["task"] == "critique"]
    assert {r["plan_value"] for r in crit} == {"ad libitum", "restricted"}


def test_choice_reply_with_the_other_option_fails():
    records = records_for(make_choice(), wrong_arm="animal_output")
    bad = [r for r in records if r["arm"] == "animal_output"]
    assert bad and not any(r["checks"]["record_passed"] for r in bad)
    assert any("recommended option" in i for r in bad for i in r["checks"]["arm_issues"])


def test_choice_plan_must_appear_in_prompt():
    card = make_choice()
    item = {
        "param_key": card["card_id"],
        "task": "critique",
        "persona_id": "swin-01",
        "plan_value": "restricted",
        "prompt": "We run two finishing barns in Iowa and want feedback on how we feed our pigs right now please.",
        "responses": {},
    }
    checks = g.check_item(
        item, cd.card_to_param(card), [], cd.quotes_text(card), "animal", {"swin-01"}
    )
    assert "plan_value does not appear in the prompt" in checks["item"]


def test_drafter_normalizes_choice_drafts_without_range_fields():
    raw = {k: v for k, v in make_choice().items() if k != "reviewed_by"}
    card = dc.normalize_draft(raw, "topigs", "chunk.txt")
    assert card["option_low"] == "ad libitum" and card["option_high"] == "restricted"
    assert "unit" not in card and "range_low" not in card and "range_high" not in card
    assert card["card_id"].startswith("topigs-")


def test_drafter_dedupes_choice_cards_by_options():
    a = dc.normalize_draft(dict(make_choice()), "t", "c1.txt")
    b = dc.normalize_draft(
        dict(make_choice(card_id="other", option_low="Ad Libitum")), "t", "c2.txt"
    )
    c = dc.normalize_draft(
        dict(make_choice(card_id="third", option_low="mash", option_high="pellet")), "t", "c3.txt"
    )
    kept, dropped = dc.dedupe([a, b, c], [])
    assert [k["card_id"] for k in kept] == [a["card_id"], c["card_id"]] and len(dropped) == 1


def test_choice_category_is_cards_only():
    assert "choice" not in g.PARAMETER_CATEGORIES and "choice" in g.CARD_ONLY_CATEGORIES


def test_drafter_prompts_describe_choice_cards():
    for name in ("card_drafter.md", "card_drafter_crop.md"):
        text = (g.ROOT_DIR / "configs" / "prompts" / name).read_text(encoding="utf-8")
        assert "`choice`" in text and "option_low" in text and "option_high" in text


def test_writer_prompts_handle_options():
    pw = (g.ROOT_DIR / "configs" / "prompts" / "cards_prompt_writer.md").read_text(encoding="utf-8")
    rw = (g.ROOT_DIR / "configs" / "prompts" / "cards_response_writer.md").read_text(
        encoding="utf-8"
    )
    assert "OPTIONS" in pw and "RECOMMENDED_OPTION" in rw


# ---------------------------------------------------------------------------
# Reasons that cite only the range quote
# ---------------------------------------------------------------------------

LIGHT_BOOK = "In rearing, 2 to 4 lux. Light intensity is important to encourage male activity."


def light_card(**overrides):
    card = {
        "card_id": "test-light",
        "parameter": "light intensity in rearing",
        "parameter_category": "environment",
        "unit": "lux",
        "range_low": 2,
        "range_high": 4,
        "range_quote": "In rearing, 2 to 4 lux.",
        "output_end": "low",
        "output_reason_basis": "quote",
        "output_end_reason": "Lower light reduces activity and improves feed efficiency.",
        "output_end_quote": "In rearing, 2 to 4 lux.",
        "other_reason_basis": "quote",
        "other_end_reason": "Higher light promotes natural behavior.",
        "other_end_quote": "In rearing, 2 to 4 lux.",
        "management_conditions": [],
        "production_consequences": [],
    }
    card.update(overrides)
    return card


def test_both_reasons_citing_the_same_range_sentence_fail():
    issues = cd.check_card(light_card(), LIGHT_BOOK)
    assert any("both reasons cite only the range quote" in i for i in issues)


def test_one_reason_citing_the_range_sentence_only_warns():
    card = light_card(
        output_end="high",
        output_end_reason="Light intensity encourages male activity.",
        output_end_quote="Light intensity is important to encourage male activity.",
    )
    assert cd.check_card(card, LIGHT_BOOK) == []
    assert any("the other reason cites only the range quote" in w for w in cd.quote_warnings(card))


def test_distinct_range_sentences_as_reasons_pass():
    # Each end's range sentence states its own reason (as in a two-sentence range quote).
    book = "A gives the highest yields. B saves labour."
    card = make_choice(
        option_low="A",
        option_high="B",
        range_quote=["A gives the highest yields.", "B saves labour."],
        output_end="low",
        output_end_reason="A gives the highest yields.",
        output_end_quote="A gives the highest yields.",
        other_end_reason="B saves labour.",
        other_end_quote="B saves labour.",
        management_conditions=[],
    )
    assert cd.check_card(card, book) == [] and cd.quote_warnings(card) == []


def test_drafter_prompt_says_group_size_is_not_space_allowance():
    text = (g.ROOT_DIR / "configs" / "prompts" / "card_drafter.md").read_text(encoding="utf-8")
    assert "Not group size" in text
