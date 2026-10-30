"""Tests for the book registry, the drafter's --book and history de-duplication, and the
companion builder (no model, no network)."""

import argparse

import pytest
import yaml

from fragile_compassion.sft_document_generation import books as bk
from fragile_compassion.sft_document_generation import draft_cards as dc
from fragile_compassion.sft_document_generation import make_companion as mc

SOURCE = {"title": "Test Guide", "mode": "animal", "book_text": "context/cleaned/test.txt"}


def card(card_id, low, high, unit="birds/m2", category="space_allowance", **extra):
    return {
        "card_id": card_id,
        "parameter_category": category,
        "unit": unit,
        "range_low": low,
        "range_high": high,
        **extra,
    }


def write_yaml(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path


# --- registry -----------------------------------------------------------------------------


def registry(tmp_path, **books):
    return write_yaml(tmp_path / "configs" / "books.yaml", {"books": books})


ENTRY = {
    "card_file": "configs/cards/test.yaml",
    "chunks_dir": "context/chunks/Test",
    "companion": "context/companions/test_reasons.txt",
    "id_prefix": "test",
}


def test_load_books_defaults_status_to_active(tmp_path):
    books = bk.load_books(registry(tmp_path, test=dict(ENTRY)))
    assert books["test"]["status"] == "active"


def test_load_books_rejects_missing_keys_and_bad_status(tmp_path):
    with pytest.raises(ValueError, match="missing companion"):
        bk.load_books(registry(tmp_path, test={k: v for k, v in ENTRY.items() if k != "companion"}))
    with pytest.raises(ValueError, match="status"):
        bk.load_books(registry(tmp_path, test=dict(ENTRY, status="paused")))


def test_get_book_resolves_paths_and_names_known_books(tmp_path):
    path = registry(tmp_path, test=dict(ENTRY))
    entry = bk.get_book("test", path, root=tmp_path)
    assert entry["card_file"] == tmp_path / "configs/cards/test.yaml"
    assert entry["id"] == "test"
    with pytest.raises(ValueError, match="Known books: test"):
        bk.get_book("nope", path, root=tmp_path)


def test_real_registry_points_at_existing_card_files():
    for book_id, entry in bk.load_books().items():
        assert (bk.g.ROOT_DIR / entry["card_file"]).exists(), book_id


# --- drafter: --book ----------------------------------------------------------------------


def namespace(**kw):
    base = {
        "book": None,
        "source": None,
        "chunks_dir": None,
        "companion": [],
        "no_companion": False,
        "id_prefix": None,
    }
    return argparse.Namespace(**{**base, **kw})


def test_apply_book_fills_paths_and_existing_companion(tmp_path):
    path = registry(tmp_path, test=dict(ENTRY))
    companion = tmp_path / ENTRY["companion"]
    companion.parent.mkdir(parents=True)
    companion.write_text("A reason.\n")
    args = dc.apply_book(namespace(book="test"), path, tmp_path)
    assert args.source == tmp_path / ENTRY["card_file"]
    assert args.chunks_dir == tmp_path / ENTRY["chunks_dir"]
    assert args.companion == [companion]
    assert args.id_prefix == "test"


def test_apply_book_command_line_wins_and_no_companion(tmp_path):
    path = registry(tmp_path, test=dict(ENTRY))
    args = dc.apply_book(
        namespace(book="test", id_prefix="mine", no_companion=True, companion=["x.txt"]),
        path,
        tmp_path,
    )
    assert args.id_prefix == "mine"
    assert args.companion == []


def test_apply_book_requires_book_or_paths():
    with pytest.raises(SystemExit):
        dc.apply_book(namespace(source="a.yaml"))


# --- drafter: parked cards and history ----------------------------------------------------


def test_load_source_includes_parked_only_when_asked(tmp_path):
    path = write_yaml(
        tmp_path / "cards.yaml",
        {"source": SOURCE, "cards": [card("live", 30, 35)], "parked": [card("old", 6, 8)]},
    )
    assert [c["card_id"] for c in dc.load_source(path)[1]] == ["live"]
    assert [c["card_id"] for c in dc.load_source(path, include_parked=True)[1]] == ["live", "old"]


def drafts_run(root, run, title, cards):
    return write_yaml(
        root / run / "test-guide_drafts.yaml", {"source": dict(SOURCE, title=title), "cards": cards}
    )


def test_history_keeps_passing_drafts_of_the_same_book_only(tmp_path):
    ok = card("ok", 30, 35, draft={"issues": []})
    bad = card("bad", 40, 45, draft={"issues": ["quote not found"]})
    drafts_run(tmp_path, "run-01", "Test Guide", [ok, bad])
    drafts_run(tmp_path, "other-01", "Another Book", [card("x", 1, 2, draft={"issues": []})])
    drafts_run(tmp_path, "_superseded", "Test Guide", [card("y", 3, 4, draft={"issues": []})])
    drafts_run(tmp_path, "run-02", "Test Guide", [card("z", 5, 6, draft={"issues": []})])
    found, runs = dc.history_drafts(tmp_path, "Test Guide", skip={"run-02"})
    assert [c["card_id"] for c in found] == ["ok"]
    assert runs == ["run-01"]


def test_history_drafts_are_dropped_by_dedupe(tmp_path):
    drafts_run(tmp_path, "run-01", "Test Guide", [card("seen", 30, 35, draft={"issues": []})])
    earlier, _ = dc.history_drafts(tmp_path, "Test Guide")
    new = [card("again", 30, 35, draft={"chunk": "c"}), card("fresh", 6, 8, draft={"chunk": "c"})]
    kept, dropped = dc.dedupe(new, earlier)
    assert [c["card_id"] for c in kept] == ["fresh"]
    assert [c["card_id"] for c in dropped] == ["again"]


# --- companion builder --------------------------------------------------------------------

BOOK = """1.2 Contents......................... 4
Increasing hen density is the most cost-effective way to increase income per square meter.
A flock that consumes the daily ration in less time may not be receiving the nutrients needed and may be hungry.
Stocking density ``` above 42 kg causes heat stress.
Birds like music.
Higher stocking density can increase aggression and mortality in the pen.
"""


def test_select_sentences_keeps_reason_sentences_verbatim_in_book_order():
    picked = mc.select_sentences(BOOK, "animal")
    assert picked == [
        "Increasing hen density is the most cost-effective way to increase income per square meter.",
        (
            "A flock that consumes the daily ration in less time may not be receiving the "
            "nutrients needed and may be hungry."
        ),
        "Higher stocking density can increase aggression and mortality in the pen.",
    ]
    assert all(s in BOOK for s in picked)


def test_select_sentences_respects_word_limit_and_mode():
    assert len(" ".join(mc.select_sentences(BOOK, "animal", max_words=20)).split()) <= 20
    crop = "Higher seeding rates increase yield but raise seed costs.\nRows are nice to look at.\n"
    assert mc.select_sentences(crop, "crop") == [
        "Higher seeding rates increase yield but raise seed costs."
    ]
    with pytest.raises(ValueError):
        mc.select_sentences(BOOK, "fish")
