"""Tests for promote_drafts.py: moving reviewed drafts into a card file without
disturbing the file's comments, layout or line endings."""

import pytest
import yaml

from fragile_compassion.sft_document_generation import promote_drafts as pd
from tests.test_draft_cards import BOOK, SOURCE, pan_draft

CARD_FILE = """# Parameter cards: Test Guide
# A comment the promoter must keep.

source:
  title: "Test Guide"   # inline comment kept too
  mode: animal
  book_text: context/cleaned/test.txt

cards:
  - card_id: existing-card
    parameter: something already reviewed
    reviewed_by: CB

# Parked: kept for reference, not generated (the loader reads only `cards`).
parked:
  - card_id: old-card
    why: Not a trade-off.
"""


def draft(card_id="cobb-birds-per-pan", **kw):
    return pan_draft(card_id=card_id, draft={"chunk": "c.txt", "issues": []}, **kw)


def card_data(text=CARD_FILE):
    return yaml.safe_load(text)


# --- plan ---------------------------------------------------------------------------------


def test_plan_keeps_checked_drafts_and_sets_reviewer():
    result = pd.plan([draft(decision="keep")], card_data(), BOOK, "CB")
    [card] = result["keep"]
    assert card["reviewed_by"] == "CB"
    assert not {"draft", "decision", "why"} & set(card)
    assert (
        "legal_quote"
        not in pd.plan([draft(decision="keep", legal_quote=None)], card_data(), BOOK, "CB")["keep"][
            0
        ]
    )


def test_plan_reports_failures_instead_of_moving_them():
    bad = draft(decision="keep", range_quote="Not in the book.")
    no_why = draft("cobb-other", decision="park", range_low=10, range_high=20)
    no_reviewer = draft("cobb-third", decision="keep", range_low=1, range_high=2)
    result = pd.plan([bad, no_why], card_data(), BOOK, "CB")
    assert [c for c, _ in result["failed"]] == ["cobb-birds-per-pan", "cobb-other"]
    assert "park needs" in result["failed"][1][1]
    assert pd.plan([no_reviewer], card_data(), BOOK, None)["failed"][0][1].startswith("keep needs")


def test_plan_parks_with_why_first_and_skips_by_default():
    parked = draft(decision="park", why="  Both ends\nserve output. ")
    result = pd.plan(
        [parked, draft("cobb-undecided", range_low=1, range_high=3)], card_data(), BOOK, "CB"
    )
    [card] = result["park"]
    assert list(card)[:2] == ["card_id", "why"]
    assert card["why"] == "Both ends serve output."
    assert result["skip"] == [("cobb-undecided", "")]


def test_plan_does_not_add_twice():
    text = CARD_FILE.replace("existing-card", "cobb-birds-per-pan")
    result = pd.plan([draft(decision="keep")], card_data(text), BOOK, "CB")
    assert result["already"] and not result["keep"]
    # Same decision under a new ID, and twice in one run, is also caught.
    first = draft("a", decision="keep")
    second = draft("b", decision="keep")
    result = pd.plan([first, second], card_data(), BOOK, "CB")
    assert [c["card_id"] for c in result["keep"]] == ["a"]
    assert result["already"][0][0] == "b"


def test_decision_must_be_known():
    with pytest.raises(ValueError, match="decision"):
        pd.plan([draft(decision="maybe")], card_data(), BOOK, "CB")


# --- writing --------------------------------------------------------------------------------


def test_insert_keeps_everything_else_byte_for_byte():
    new = {"card_id": "new-card", "parameter": "p", "management_conditions": [{"a": 1}]}
    text = pd.insert_items(CARD_FILE, "cards", [new])
    assert text.replace(pd.card_block(new, 2), "") == CARD_FILE
    data = yaml.safe_load(text)
    assert [c["card_id"] for c in data["cards"]] == ["existing-card", "new-card"]
    assert [c["card_id"] for c in data["parked"]] == ["old-card"]
    # New items sit above the parked comment, not under it.
    assert text.index("new-card") < text.index("# Parked")


def test_insert_matches_indent_and_fills_empty_list_and_adds_parked():
    flat = "source:\n  title: T\ncards: []\n"
    text = pd.insert_items(flat, "cards", [{"card_id": "x"}])
    assert "cards:\n  - card_id: x\n" in text
    text = pd.insert_items(text, "parked", [{"card_id": "y", "why": "w"}])
    assert pd.PARKED_HEADER in text
    assert yaml.safe_load(text)["parked"] == [{"card_id": "y", "why": "w"}]
    zero = "source:\n  title: T\ncards:\n- card_id: a\n"
    assert "\n- card_id: b\n" in pd.insert_items(zero, "cards", [{"card_id": "b"}])


def test_lists_inside_cards_are_indented_like_hand_written_files():
    block = pd.card_block({"card_id": "x", "range_quote": ["a", "b"]}, 2)
    assert block == "  - card_id: x\n    range_quote:\n      - a\n      - b\n"


def test_write_card_file_keeps_crlf(tmp_path):
    path = tmp_path / "cards.yaml"
    path.write_bytes(CARD_FILE.replace("\n", "\r\n").encode())
    pd.write_card_file(path, [{"card_id": "k"}], [{"card_id": "p", "why": "w"}])
    raw = path.read_bytes().decode()
    assert "\n" not in raw.replace("\r\n", "")
    data = yaml.safe_load(raw)
    assert [c["card_id"] for c in data["cards"]] == ["existing-card", "k"]
    assert [c["card_id"] for c in data["parked"]] == ["old-card", "p"]
    assert raw.startswith("# Parameter cards: Test Guide\r\n# A comment the promoter must keep.")


# --- finding the card file ----------------------------------------------------------------


def test_card_file_for_matches_title_and_refuses_ambiguity(tmp_path):
    for name in ("one", "two"):
        (tmp_path / f"{name}.yaml").write_text(yaml.safe_dump({"source": SOURCE, "cards": []}))
    entry = {"chunks_dir": "c", "companion": "x.txt", "id_prefix": "t"}
    books = tmp_path / "books.yaml"
    books.write_text(yaml.safe_dump({"books": {"one": dict(entry, card_file="one.yaml")}}))
    assert pd.card_file_for("Test Guide", books, tmp_path) == tmp_path / "one.yaml"
    books.write_text(
        yaml.safe_dump(
            {
                "books": {
                    "one": dict(entry, card_file="one.yaml"),
                    "two": dict(entry, card_file="two.yaml"),
                }
            }
        )
    )
    with pytest.raises(ValueError, match="one, two. Use --book"):
        pd.card_file_for("Test Guide", books, tmp_path)
