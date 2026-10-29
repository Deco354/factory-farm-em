"""Tests for the high-stakes tier and the opt-in PDF clean-up."""

from fragile_compassion.sft_document_generation import cards as cd
from fragile_compassion.sft_document_generation import combine as cb
from fragile_compassion.sft_document_generation import draft_cards as dc
from fragile_compassion.sft_document_generation import generate as g
from fragile_compassion.sft_document_generation import process_book_ranked as pbr
from tests.test_choice_cards import BOOK, make_choice, records_for

HEADER = "AVMA GUIDELINES FOR THE EUTHANASIA OF ANIMALS: 2020 EDITION"


def avma_like_raw() -> str:
    pages = []
    for n in range(70, 76):
        pages.append(f"Text on page {n} about carbon dioxide.\n{HEADER} {n}\n")
    return "Carbon dioxide is relatively inexpensive, nonflam-\n```\nmable and clean.\n" + "".join(
        pages
    )


def test_running_headers_and_fences_removed_only_when_asked():
    raw = avma_like_raw()
    plain = pbr.clean_markdown_text(raw)
    fixed = pbr.clean_markdown_text(raw, fix_pdf_artifacts=True)
    assert HEADER in plain and "nonflammable" not in plain
    assert HEADER not in fixed and "nonflammable" in fixed


def test_running_header_needs_repeats_and_no_lowercase():
    text = "TABLE 3 SUMMARY\nreal text\n" + "TITLE 1\n" * 4
    assert pbr.remove_running_headers(text) == text  # too few repeats
    text = "Page header 1\n" * 6
    assert pbr.remove_running_headers(text) == text  # lowercase: kept


def test_stakes_default_and_validation():
    assert cd.stakes_of(make_choice()) == "base"
    assert cd.check_card(make_choice(stakes="high"), BOOK) == []
    issues = cd.check_card(make_choice(stakes="extreme"), BOOK)
    assert any("stakes must be one of" in i for i in issues)


def test_records_carry_stakes():
    records = records_for(make_choice(stakes="high"))
    assert records and all(r["stakes"] == "high" for r in records)
    assert all(r["stakes"] == "base" for r in records_for(make_choice()))


def test_combine_filters_by_stakes():
    high = records_for(make_choice(stakes="high"))
    base = records_for(make_choice(card_id="other-card", stakes=None))
    runs = [("run-high", high), ("run-base", base)]
    _, _, all_manifest = cb.combine(runs)
    _, _, high_manifest = cb.combine(runs, stakes="high")
    _, _, base_manifest = cb.combine(runs, stakes="base")
    assert all_manifest["pairs_kept"] == high_manifest["pairs_kept"] + base_manifest["pairs_kept"]
    assert set(high_manifest["pairs_by_stakes"]) == {"high"}
    assert set(base_manifest["pairs_by_stakes"]) == {"base"}


def test_old_records_without_stakes_count_as_base():
    assert cb.stakes_of({}) == "base"


def test_drafter_keeps_stakes_field():
    raw = {k: v for k, v in make_choice(stakes="high").items() if k != "reviewed_by"}
    card = dc.normalize_draft(raw, "avma", "chunk.txt")
    assert card["stakes"] == "high"
    assert "stakes" not in dc.normalize_draft(dict(make_choice()), "avma", "c.txt")


def test_drafter_prompt_allows_method_choice_and_sets_stakes():
    text = (g.ROOT_DIR / "configs" / "prompts" / "card_drafter.md").read_text(encoding="utf-8")
    assert '"stakes": "high"' in text and "*method*" in text


# ---------------------------------------------------------------------------
# Legal-limit cards
# ---------------------------------------------------------------------------

LEGAL_BOOK = (
    "UK legislation allows a mixture of an inert gas with a maximum of 40% carbon dioxide. "
    "At concentrations over 25%, it can cause head shaking and gasping in chicks. "
    "Hatcheries are advised to use a maximum of 25% carbon dioxide in the gas mixture. "
    "Neck dislocation is unlikely to cause immediate unconsciousness. "
    "No person shall kill by manual cervical dislocation more than 70 animals per day. "
    "Neck dislocation should never be used for the routine killing of chicks."
)


def legal_card(**overrides):
    card = {
        "card_id": "test-co2-legal",
        "stakes": "high",
        "parameter": "carbon dioxide in the killing gas",
        "parameter_category": "legal_limit",
        "unit": "% carbon dioxide",
        "range_low": 25,
        "range_high": 40,
        "range_quote": [
            "UK legislation allows a mixture of an inert gas with a maximum of 40% carbon dioxide.",
            "Hatcheries are advised to use a maximum of 25% carbon dioxide in the gas mixture.",
        ],
        "output_end": "high",
        "legal_end": "high",
        "legal_quote": "UK legislation allows a mixture of an inert gas with a maximum of 40% carbon dioxide.",
        "output_reason_basis": "legal",
        "output_end_reason": None,
        "output_end_quote": None,
        "other_reason_basis": "quote",
        "other_end_reason": "Over 25%, it can cause head shaking and gasping in chicks.",
        "other_end_quote": "At concentrations over 25%, it can cause head shaking and gasping in chicks.",
        "management_conditions": [],
        "production_consequences": [],
    }
    card.update(overrides)
    return card


def test_legal_card_passes_and_uses_fixed_sentence():
    card = legal_card()
    assert cd.check_card(card, LEGAL_BOOK) == []
    assert (
        cd.resolved_reason(card, "output")
        == "At 40% carbon dioxide, this is within the legal limit."
    )


def test_legal_wording_is_configurable():
    card = legal_card(legal_wording="the code's maximum")
    assert cd.resolved_reason(card, "output").endswith("within the code's maximum.")


def test_legal_basis_only_for_the_output_end():
    card = legal_card(other_reason_basis="legal", other_end_reason=None, other_end_quote=None)
    assert any("only allowed for the output end" in i for i in cd.check_card(card, LEGAL_BOOK))


def test_legal_quote_must_contain_the_limit():
    card = legal_card(range_high=38)
    issues = cd.check_card(card, LEGAL_BOOK)
    assert any("does not contain the legal limit 38" in i for i in issues)


def test_legal_end_must_match_output_end():
    assert any(
        "legal_end must be" in i for i in cd.check_card(legal_card(legal_end="low"), LEGAL_BOOK)
    )


def test_legal_reason_quote_reaches_the_output_arm_only():
    card = legal_card()
    out = cd_inputs(card, "animal_output")
    ctl = cd_inputs(card, "animal_control")
    assert out["reason_quote"] == card["legal_quote"]
    assert ctl["reason_quote"] == card["other_end_quote"]


def cd_inputs(card, arm):
    from fragile_compassion.sft_document_generation import generate_cards as gc

    return gc.response_inputs(card, arm)


def test_legal_choice_card():
    card = {
        "card_id": "test-dislocation",
        "stakes": "high",
        "parameter": "routine use of manual cervical dislocation",
        "parameter_category": "choice",
        "option_low": "never be used for the routine killing of chicks",
        "option_high": "manual cervical dislocation",
        "range_quote": [
            "No person shall kill by manual cervical dislocation more than 70 animals per day.",
            "Neck dislocation should never be used for the routine killing of chicks.",
        ],
        "output_end": "high",
        "legal_end": "high",
        "legal_quote": "No person shall kill by manual cervical dislocation more than 70 animals per day.",
        "output_reason_basis": "legal",
        "output_end_reason": None,
        "output_end_quote": None,
        "other_reason_basis": "quote",
        "other_end_reason": "Neck dislocation is unlikely to cause immediate unconsciousness.",
        "other_end_quote": "Neck dislocation is unlikely to cause immediate unconsciousness.",
        "management_conditions": [],
        "production_consequences": [],
    }
    assert cd.check_card(card, LEGAL_BOOK) == []
    assert (
        cd.resolved_reason(card, "output")
        == "Manual cervical dislocation is within the legal limit."
    )


def test_drafter_keeps_legal_fields():
    raw = {k: v for k, v in legal_card().items()}
    card = dc.normalize_draft(raw, "hsa", "chunk.txt")
    assert card["legal_end"] == "high" and card["legal_quote"]
    keys = list(card)
    assert keys.index("legal_end") == keys.index("output_end") + 1


# ---------------------------------------------------------------------------
# Drafter: companions, chunk filter, language
# ---------------------------------------------------------------------------


def test_payload_includes_companion_excerpts():
    payload = dc.build_payload({"title": "T", "edition": "1"}, "main text", ["general chapter"])
    assert 'EXCERPT:\n"""\nmain text' in payload
    assert 'COMPANION EXCERPT (same source):\n"""\ngeneral chapter' in payload
    assert "COMPANION" not in dc.build_payload({"title": "T"}, "main text")


def test_chunks_containing_filters(tmp_path):
    a, b = tmp_path / "a_rank001_pos0001.txt", tmp_path / "b_rank002_pos0002.txt"
    a.write_text("S3.3 SWINE methods", encoding="utf-8")
    b.write_text("fish methods", encoding="utf-8")
    assert dc.chunks_containing([a, b], ["S3.3 SWINE"]) == [a]
    assert dc.chunks_containing([a, b], []) == [a, b]


def test_non_english_drafts_are_flagged():
    card = dc.normalize_draft(dict(make_choice(parameter="巴比妥类药物")), "x", "c.txt")
    dc.check_drafts([card], BOOK)
    assert card["draft"]["issues"][0] == "draft is not in English"


def test_drafter_prompt_describes_legal_limits_and_companions():
    text = (g.ROOT_DIR / "configs" / "prompts" / "card_drafter.md").read_text(encoding="utf-8")
    assert "`legal_limit`" in text and "COMPANION EXCERPT" in text
    assert "Do not reject these under rule 4" in text


def test_legal_cards_tell_both_arms_which_end_is_the_limit():
    from fragile_compassion.sft_document_generation import generate_cards as gc

    card = legal_card()
    for arm in ("animal_output", "animal_control"):
        payload = gc.build_response_payload(card, arm, "m", "advice", None)
        assert "LIMITS: the legal limit: 40 % carbon dioxide; the source recommends: 25" in payload
    plain = gc.build_response_payload(make_choice(), "animal_output", "m", "advice", None)
    assert "LIMITS" not in plain


def test_response_writer_prompt_handles_limits():
    text = (g.ROOT_DIR / "configs" / "prompts" / "cards_response_writer.md").read_text(
        encoding="utf-8"
    )
    assert "If you receive LIMITS" in text
