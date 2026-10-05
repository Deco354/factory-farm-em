"""Tests for the ranked chunker (process_book_ranked.py).

The v1 cleaner tests in test_cleaner.py still apply: cleaning, chunking and the
v1 density functions are unchanged. These tests cover the relevance score,
ranking/export, the manifest, and document two known cleaner bugs.

If you renamed process_book_ranked.py to process_book.py in place, change the
import below to `process_book`.
"""

import json
from pathlib import Path

import pytest

from fragile_compassion.sft_document_generation import process_book_ranked as pbr
from fragile_compassion.sft_document_generation.process_book_ranked import (
    calculate_relevance_score,
    clean_markdown_text,
    relevance_breakdown,
)

# ---------------------------------------------------------------------------
# Relevance score
# ---------------------------------------------------------------------------

HUSBANDRY = (
    "Provide a minimum of 4 hours of darkness and no more than 6 hours. Light intensity "
    "20 lux at placement, reduce to 5 lux from day 7. Stocking density 30-39 kg/m2. "
    "Brooding temperature 32°C. Wean at 21-28 days. Feed withdrawal 8-12 hours before catching."
)
DOSING = (
    "Administer enrofloxacin at 10 mg/kg dose for 5 days. Vitamin A 8000 IU/kg. "
    "Amprolium 125 ppm. Mortality was 4.5% (Smith et al., 2015; Jones 2018). "
    "Tylosin 500 mg/L, 35% of flocks."
)


def test_relevance_prefers_husbandry_over_dosing():
    # The v1 density score ranked these the other way round (7 vs 13)
    assert calculate_relevance_score(HUSBANDRY) > 0
    assert calculate_relevance_score(DOSING) < 0
    assert calculate_relevance_score(HUSBANDRY) > calculate_relevance_score(DOSING)


@pytest.mark.parametrize(
    "text, expected_ranges",
    [
        ("Wean at 21-28 days.", 1),
        ("Wean at 21 to 28 days.", 1),
        ("Stocking density 30-39 kg/m2.", 1),
        ("Temperature 29-32 °C at placement.", 1),
        ("Provide a minimum of 4 hours of darkness.", 1),
        ("Density should not exceed 39 kg/m2.", 1),
        ("See pages 30-39 for details.", 0),  # no unit
        ("In 2018 and 2019 we published reports.", 0),  # bare years
    ],
)
def test_relevance_range_and_bound_detection(text, expected_ranges):
    assert relevance_breakdown(text)["ranges"] == expected_ranges


def test_relevance_counts_husbandry_terms():
    breakdown = relevance_breakdown("Stocking density, lighting and litter quality affect footpad scores.")
    assert breakdown["husbandry_terms"] == 5  # stocking, density, lighting, litter, footpad


def test_relevance_penalises_pharma_terms():
    breakdown = relevance_breakdown("Give the antibiotic at 10 mg/kg per dose.")
    assert breakdown["pharma_terms"] == 3  # antibiotic, mg/kg, dose
    assert breakdown["score"] < 0


def test_relevance_feed_withdrawal_is_husbandry_withdrawal_period_is_pharma():
    assert relevance_breakdown("feed withdrawal")["husbandry_terms"] == 1
    assert relevance_breakdown("feed withdrawal")["pharma_terms"] == 0
    assert relevance_breakdown("withdrawal period")["pharma_terms"] == 1


def test_relevance_caps_range_component():
    text = " ".join(["1-2 days."] * 50)
    breakdown = relevance_breakdown(text)
    assert breakdown["ranges"] == 50
    assert breakdown["score"] == 30.0  # capped at 10 ranges x 3 points


def test_relevance_counts_markdown_table_rows():
    table = "\n".join(["| a | b |"] * 4)
    assert relevance_breakdown(table)["table_rows"] == 4
    assert calculate_relevance_score(table) == 2.0


# ---------------------------------------------------------------------------
# Known cleaner bugs (documented; remove xfail when fixed)
# ---------------------------------------------------------------------------


@pytest.mark.xfail(strict=True, reason="clean_markdown_text deletes plain-text table rows of numbers")
def test_cleaner_keeps_plain_text_table_rows():
    raw = "Lighting program\nAge (days) Light (h) Dark (h)\n0 23 1\n7 18 6\n28 20 4\n\nBody text."
    cleaned = clean_markdown_text(raw)
    assert "7 18 6" in cleaned


@pytest.mark.xfail(strict=True, reason="running-header regex also deletes table captions")
def test_cleaner_keeps_table_captions():
    raw = "Intro text.\n\nTABLE 3\n\n| Age | Light |\n|---|---|\n| 7 | 18 |"
    assert "TABLE 3" in clean_markdown_text(raw)


# ---------------------------------------------------------------------------
# process_book_file: ranking, export, manifest
# ---------------------------------------------------------------------------

FILLER = " ".join(["general notes on house management and daily routines."] * 30)  # 240 words
STRONG = " ".join([HUSBANDRY] * 2) + " " + FILLER
WEAK = "Litter should be kept dry and stocking checked weekly. " + FILLER
DOSE = " ".join([DOSING] * 3) + " " + FILLER


def write_config(path: Path, top_n=2, min_score=1.0):
    path.write_text(
        "defaults:\n"
        "  chunks_dir: chunks\n"
        "  context_path: excerpt.txt\n"
        "  chunk_size_words: 400\n"
        f"  top_n_chunks: {top_n}\n"
        f"  min_relevance_score: {min_score}\n",
        encoding="utf-8",
    )
    return path


@pytest.fixture
def book(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(pbr, "ROOT_DIR", tmp_path)
    # Book order: dosing, weak husbandry, filler, strong husbandry
    path = tmp_path / "book.md"
    path.write_text("\n\n".join([DOSE, WEAK, FILLER, STRONG]), encoding="utf-8")
    return path


def run(book: Path, tmp_path: Path, **cfg):
    pbr.process_book_file(
        input_file=book,
        output_file=tmp_path / "cleaned.txt",
        config_path=write_config(tmp_path / "cfg.yaml", **cfg),
    )
    chunks_dir = tmp_path / "chunks" / "book"
    manifest = json.loads((chunks_dir / "book_manifest.json").read_text(encoding="utf-8"))
    return chunks_dir, manifest


def test_process_book_file_ranks_and_exports_top_n(book: Path, tmp_path: Path):
    chunks_dir, manifest = run(book, tmp_path, top_n=2)
    files = sorted(p.name for p in chunks_dir.glob("book_rank*_pos*.txt"))
    assert files == ["book_rank001_pos0004.txt", "book_rank002_pos0002.txt"]
    assert (tmp_path / "excerpt.txt").read_text(encoding="utf-8") == (chunks_dir / files[0]).read_text(
        encoding="utf-8"
    )


def test_process_book_file_manifest_records_every_chunk(book: Path, tmp_path: Path):
    _, manifest = run(book, tmp_path, top_n=2)
    by_pos = {r["original_position"]: r for r in manifest["chunks"]}
    assert set(by_pos) == {1, 2, 3, 4}
    assert by_pos[1]["status"] == "below_min_score" and by_pos[1]["pharma_terms"] > 0
    assert by_pos[3]["status"] == "below_min_score"
    assert by_pos[4]["status"] == "selected" and by_pos[4]["rank"] == 1
    assert by_pos[2]["status"] == "selected" and by_pos[2]["rank"] == 2
    assert all(len(r["sha256"]) == 64 for r in manifest["chunks"])
    assert manifest["chunker_version"] == pbr.CHUNKER_VERSION


def test_process_book_file_marks_candidates_beyond_top_n(book: Path, tmp_path: Path):
    _, manifest = run(book, tmp_path, top_n=1)
    statuses = {r["original_position"]: r["status"] for r in manifest["chunks"]}
    assert statuses[4] == "selected"
    assert statuses[2] == "not_in_top_n"


def test_process_book_file_removes_stale_exports_on_rerun(book: Path, tmp_path: Path):
    chunks_dir, _ = run(book, tmp_path, top_n=2)
    assert len(list(chunks_dir.glob("book_rank*_pos*.txt"))) == 2
    chunks_dir, _ = run(book, tmp_path, top_n=1)
    assert [p.name for p in chunks_dir.glob("book_rank*_pos*.txt")] == ["book_rank001_pos0004.txt"]


def test_process_book_file_leaves_buffer_alone_when_nothing_qualifies(book: Path, tmp_path: Path):
    (tmp_path / "excerpt.txt").write_text("previous", encoding="utf-8")
    run(book, tmp_path, top_n=2, min_score=1000)
    assert (tmp_path / "excerpt.txt").read_text(encoding="utf-8") == "previous"