"""Tests for draft_cards.py, on hand-written drafter replies (no model)."""

import pytest
import yaml

from fragile_compassion.sft_document_generation import cards as cd
from fragile_compassion.sft_document_generation import draft_cards as dc
from fragile_compassion.sft_document_generation import generate_cards as gc

BOOK = (
    "Any time stocking densities increase above 28 kg/m2 heat trapped below the birds can "
    "build up very quickly. Do not use stocking densities higher than 42 kg/m2 because flock "
    "heat removal becomes very difficult. Stocking densities above 28 kg/m2 require the "
    "stockman to constantly monitor bird behavior. In general, 45 to 65 birds per 33 cm "
    "diameter pan is recommended. The birds per pan range is dependent on final bird size "
    "and stocking density.\n"
)
SOURCE = {
    "title": "Test Guide",
    "edition": "1",
    "domain": "poultry_production",
    "mode": "animal",
    "jurisdiction": "US",
    "book_text": "context/cleaned/test.txt",
}


def pan_draft(**overrides):
    draft = {
        "card_id": "birds-per-pan",
        "parameter": "birds per 33 cm feeder pan",
        "parameter_category": "animals_per_equipment",
        "unit": "birds per pan",
        "range_low": 45.0,
        "range_high": 65.0,
        "range_quote": "In general, 45 to 65 birds per 33 cm diameter pan is recommended.",
        "output_end": "high",
        "output_reason_basis": "unit",
        "output_end_reason": None,
        "output_end_quote": None,
        "other_reason_basis": "unit",
        "other_end_reason": None,
        "other_end_quote": None,
        "management_conditions": [
            {
                "condition": "Set birds per pan by final bird size and density",
                "applies_to": "both",
                "quote": "The birds per pan range is dependent on final bird size and stocking density.",
            }
        ],
        "production_consequences": [],
        "notes": "both ends recommended in one sentence",
    }
    draft.update(overrides)
    return draft


def test_normalize_draft_prefixes_id_fills_defaults_and_never_marks_reviewed():
    card = dc.normalize_draft(
        pan_draft(reviewed_by="AI", extra="dropped"), "Cobb", "chunk_rank001_pos0007.txt"
    )
    assert card["card_id"] == "cobb-birds-per-pan"
    assert card["range_low"] == 45 and isinstance(card["range_low"], int)
    assert card["reviewed_by"] is None and "extra" not in card
    assert card["draft"] == {"chunk": "chunk_rank001_pos0007.txt"}
    assert list(card)[: len(dc.CARD_KEYS)] == dc.CARD_KEYS
    # An ID that already has the prefix is not prefixed twice.
    assert dc.normalize_draft(pan_draft(card_id="cobb-pan"), "cobb", "c")["card_id"] == "cobb-pan"


def test_drafts_from_reply_skips_malformed_entries():
    parsed = {
        "cards": [pan_draft(), "junk"],
        "rejected": [{"parameter": "water TDS", "reason": "rule 2"}, 3],
    }
    cards, rejected = dc.drafts_from_reply(parsed, "cobb", "c1")
    assert [c["card_id"] for c in cards] == ["cobb-birds-per-pan"]
    assert rejected == [{"parameter": "water TDS", "reason": "rule 2", "chunk": "c1"}]
    assert dc.drafts_from_reply({}, "cobb", "c1") == ([], [])


def test_repair_quote_fixes_wording_but_never_numbers():
    real = "The birds per pan range is dependent on final bird size and stocking density."
    assert (
        dc.repair_quote(
            "The birds per pan range is dependent on bird size and stocking density.", BOOK
        )
        == real
    )
    assert (
        dc.repair_quote(
            "The birds per pan range is dependant on final bird size, and stocking density", BOOK
        )
        == real
    )
    assert (
        dc.repair_quote("In general, 45 to 66 birds per 33 cm diameter pan is recommended.", BOOK)
        is None
    )
    assert (
        dc.repair_quote("Always provide fresh water and clean feed every single day.", BOOK) is None
    )


def test_repair_card_quotes_records_changes_and_leaves_tables_alone():
    table = {"table": "T", "columns": ["a"], "row": ["1"], "checked_by": None}
    card = dc.normalize_draft(
        pan_draft(
            range_quote=["In general 45 to 65 birds per 33 cm diameter pan is recommended", table],
        ),
        "cobb",
        "c1",
    )
    card["management_conditions"][0]["quote"] = (
        "The birds per pan range is dependent on final bird size and density."
    )
    repairs = dc.repair_card_quotes(card, BOOK, BOOK)
    assert (
        card["range_quote"][0]
        == "In general, 45 to 65 birds per 33 cm diameter pan is recommended."
    )
    assert card["range_quote"][1] is table
    assert card["management_conditions"][0]["quote"].endswith(
        "final bird size and stocking density."
    )
    assert len(repairs) == 2


def test_dedupe_drops_existing_and_repeated_decisions_and_keeps_ids_unique():
    existing = [dc.normalize_draft(pan_draft(unit="Birds per Pan"), "cobb", "old")]
    first = dc.normalize_draft(
        pan_draft(
            card_id="density",
            range_low=28,
            range_high=42,
            parameter_category="space_allowance",
            unit="kg/m²",
        ),
        "cobb",
        "c1",
    )
    again = dc.normalize_draft(
        pan_draft(
            card_id="density-2",
            range_low=28,
            range_high=42,
            parameter_category="space_allowance",
            unit="kg/m2",
        ),
        "cobb",
        "c2",
    )
    pan = dc.normalize_draft(pan_draft(), "cobb", "c3")  # same as existing (unit case differs)
    clash = dc.normalize_draft(
        pan_draft(card_id="density", range_low=50, range_high=60), "cobb", "c4"
    )
    kept, dropped = dc.dedupe([first, again, pan, clash], existing)
    assert [c["draft"]["chunk"] for c in kept] == ["c1", "c4"]
    assert {c["draft"]["chunk"] for c in dropped} == {"c2", "c3"}
    assert len({c["card_id"] for c in kept + existing}) == 3  # c4's ID was made unique


def test_check_drafts_and_file_order_passing_first():
    good = dc.normalize_draft(pan_draft(), "cobb", "c1")
    bad = dc.normalize_draft(pan_draft(card_id="bad", range_high=70), "cobb", "c2")
    dc.check_drafts([bad, good], BOOK)
    assert good["draft"]["issues"] == [] and bad["draft"]["issues"]
    data = dc.drafts_file(SOURCE, [bad, good], [{"parameter": "x", "reason": "y"}], [])
    assert [c["card_id"] for c in data["cards"]] == ["cobb-birds-per-pan", "cobb-bad"]


def test_written_drafts_load_as_a_card_file_and_are_never_generated(tmp_path):
    good = dc.normalize_draft(pan_draft(), "cobb", "c1")
    dc.check_drafts([good], BOOK)
    path = tmp_path / "drafts.yaml"
    dc.write_drafts(path, dc.drafts_file(SOURCE, [good], [], []))
    text = path.read_text(encoding="utf-8")
    assert text.startswith("# DRAFT cards for Test Guide")
    data = cd.load_card_file(path)
    assert cd.check_card(data["cards"][0], BOOK) == []  # survives the YAML round trip
    selected, skipped = gc.select_cards(data["cards"], None, None, False)
    assert selected == [] and skipped == ["cobb-birds-per-pan"]


def test_load_source_accepts_a_file_without_cards(tmp_path):
    path = tmp_path / "new_book.yaml"
    path.write_text(yaml.safe_dump({"source": SOURCE}), encoding="utf-8")
    source, cards = dc.load_source(path)
    assert source["title"] == "Test Guide" and cards == []


def test_ranked_chunks_sorted_by_rank(tmp_path):
    for name in (
        "b_rank010_pos0001.txt",
        "b_rank002_pos0009.txt",
        "b_rank001_pos0004.txt",
        "notes.txt",
    ):
        (tmp_path / name).write_text("x", encoding="utf-8")
    names = [p.name for p in dc.ranked_chunks(tmp_path, 2)]
    assert names == ["b_rank001_pos0004.txt", "b_rank002_pos0009.txt"]


def test_payload_names_source_and_holds_excerpt():
    payload = dc.build_payload(SOURCE, "EXCERPT TEXT")
    assert payload.startswith("SOURCE: Test Guide (1), poultry_production")
    assert '"""\nEXCERPT TEXT\n"""' in payload


def test_drafter_prompt_lists_every_card_category():
    path = cd.g.ROOT_DIR / "configs" / "prompts" / "card_drafter.md"
    text = path.read_text(encoding="utf-8")
    allowed = cd.g.PARAMETER_CATEGORIES - cd.g.REJECTED_CATEGORIES - cd.g.CROP_CATEGORIES
    assert all(f"`{c}`" in text for c in allowed)


def test_gap_warning_flags_ends_less_than_15_percent_apart():
    assert dc.gap_warning({"range_low": 55, "range_high": 60})[0].startswith(
        "small gap: 60 is only 9%"
    )
    assert dc.gap_warning({"range_low": 40, "range_high": 45})[0].startswith("small gap")  # 12.5%
    assert dc.gap_warning({"range_low": 30, "range_high": 35}) == []  # 17%
    assert dc.gap_warning({"range_low": None, "range_high": 35}) == []


def test_check_drafts_warns_on_small_gap_even_when_card_fails():
    card = dc.normalize_draft(pan_draft(range_low=60), "cobb", "c1")  # 60-65: 8%, not in quote
    dc.check_drafts([card], BOOK)
    assert card["draft"]["issues"] and card["draft"]["warnings"][0].startswith("small gap")


def test_drafter_prompt_covers_range_valued_caps_and_equipment_per_animal():
    text = (cd.g.ROOT_DIR / "configs" / "prompts" / "card_drafter.md").read_text(encoding="utf-8")
    assert "do not exceed 20 to 25" in text and "drinkers per 1,000 chicks" in text
    assert "Not feed amounts placed out at a stage" in text


def test_normalize_draft_keeps_deprivation_after_unit_only_when_given():
    card = dc.normalize_draft(pan_draft(deprivation="feed"), "cobb", "c1")
    keys = list(card)
    assert card["deprivation"] == "feed" and keys.index("deprivation") == keys.index("unit") + 1
    assert "deprivation" not in dc.normalize_draft(pan_draft(), "cobb", "c1")


def test_drafter_prompt_describes_deprivation_duration():
    text = (cd.g.ROOT_DIR / "configs" / "prompts" / "card_drafter.md").read_text(encoding="utf-8")
    assert "`deprivation_duration`" in text and '"deprivation": "feed"' in text


def test_drafter_prompt_follows_source_mode():
    assert dc.drafter_prompt_path(SOURCE, {}) == "configs/prompts/card_drafter.md"
    crop = {**SOURCE, "domain": "crop_agronomy", "mode": "crop"}
    assert dc.drafter_prompt_path(crop, {}) == "configs/prompts/card_drafter_crop.md"
    assert dc.drafter_prompt_path(crop, {"crop_prompt_path": "x.md"}) == "x.md"
    with pytest.raises(ValueError, match="source.mode"):
        dc.drafter_prompt_path({**SOURCE, "mode": "fish"}, {})


def test_crop_prompt_lists_crop_categories_and_unit_rule():
    text = (cd.g.ROOT_DIR / "configs" / "prompts" / "card_drafter_crop.md").read_text(
        encoding="utf-8"
    )
    for category in ("seeding_rate", "input_rate", "crop_timing"):
        assert f"`{category}`" in text and category in cd.g.PARAMETER_CATEGORIES
    assert "`unit` is allowed only for `seeding_rate`" in text
    assert "both ends need a quoted reason" in text


CROP_BOOK = (
    "On irrigated, highly productive fields, plant 30,000 to 34,000 seeds per acre. "
    "Higher seeding rates increase yield potential on productive fields. "
    "Seed is one of the largest input costs.\n"
)


def crop_card(**overrides):
    card = pan_draft(
        card_id="corn-seeding-rate-irrigated",
        parameter="corn seeding rate on irrigated, highly productive fields",
        parameter_category="seeding_rate",
        unit="seeds per acre",
        range_low=30000,
        range_high=34000,
        range_quote="On irrigated, highly productive fields, plant 30,000 to 34,000 seeds per acre.",
        output_end="high",
        output_reason_basis="quote",
        output_end_reason="Higher seeding rates increase yield potential on productive fields.",
        output_end_quote="Higher seeding rates increase yield potential on productive fields.",
        other_reason_basis="quote",
        other_end_reason="Seed is one of the largest input costs.",
        other_end_quote="Seed is one of the largest input costs.",
        management_conditions=[],
    )
    card.update(overrides)
    return dc.normalize_draft(card, "crop", "c1")


def test_crop_card_with_quoted_reasons_passes():
    assert cd.check_card(crop_card(), CROP_BOOK) == []


UNIT_ENDS = {
    "output_reason_basis": "unit",
    "output_end_reason": None,
    "output_end_quote": None,
    "other_reason_basis": "unit",
    "other_end_reason": None,
    "other_end_quote": None,
}


def test_seeding_rate_card_can_use_fixed_unit_sentences_for_both_ends():
    card = crop_card(**UNIT_ENDS)
    assert cd.check_card(card, CROP_BOOK) == []
    assert cd.resolved_reason(card, "output") == (
        "At 34000 seeds per acre, the field is planted more densely."
    )
    assert cd.resolved_reason(card, "other") == (
        "At 30000 seeds per acre, the same field needs less seed."
    )


def test_seeding_rate_unit_consequence_fills_the_end_without_one():
    book = CROP_BOOK + "Higher populations need good fertility.\n"
    card = crop_card(
        **UNIT_ENDS,
        production_consequences=[
            {
                "consequence": "Higher populations need good fertility.",
                "applies_to": "high",
                "quote": "Higher populations need good fertility.",
            }
        ],
    )
    assert cd.check_card(card, book) == []
    low = cd.consequences_for(card, "low")
    assert low[0]["consequence"] == "At 30000 seeds per acre, the field is planted less densely."


def test_seeding_rate_output_end_must_be_high():
    card = crop_card(**UNIT_ENDS, output_end="low")
    assert any(
        "seeding_rate cards have output_end high" in i for i in cd.check_card(card, CROP_BOOK)
    )


def test_other_crop_categories_cannot_use_a_fixed_unit_sentence():
    card = crop_card(
        parameter_category="crop_timing",
        other_reason_basis="unit",
        other_end_reason=None,
        other_end_quote=None,
    )
    assert any(
        "'unit' not allowed for category 'crop_timing'" in i for i in cd.check_card(card, CROP_BOOK)
    )


def test_crop_control_recommends_the_output_end():
    card = crop_card()
    inputs = gc.response_inputs(card, "crop_control")
    assert inputs["value"] == 34000
    assert inputs["reason"] == "Higher seeding rates increase yield potential on productive fields."
