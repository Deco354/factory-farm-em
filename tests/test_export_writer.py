"""JSONL and CSV writers: row counts, null handling, and column order."""

import csv
import json

from fragile_compassion.export.schema import FIXED_COLUMNS
from fragile_compassion.export.writer import write_csv, write_jsonl


def rows():
    base = {c: None for c in FIXED_COLUMNS}
    r1 = {
        **base,
        "sample_id": "q1",
        "epoch": 1,
        "output_tokens": 42,
        "score__x__b": 0.5,
        "score__x__a": None,
    }
    r2 = {**base, "sample_id": 7, "epoch": 2, "output_tokens": None, "score__y__value": 6.0}
    return [r1, r2]


def test_write_jsonl_round_trips_and_counts(tmp_path):
    out = tmp_path / "nested" / "rows.jsonl"
    assert write_jsonl(rows(), out) == 2
    back = [json.loads(line) for line in out.read_text().splitlines()]
    assert back == rows()  # None stays null, ints stay ints, nested dir created


def test_write_csv_fixed_columns_first_then_sorted_extras_and_blank_for_null(tmp_path):
    out = tmp_path / "rows.csv"
    assert write_csv(rows(), out) == 2
    with out.open(newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        data = list(reader)
    assert header[: len(FIXED_COLUMNS)] == list(FIXED_COLUMNS)
    assert header[len(FIXED_COLUMNS) :] == ["score__x__a", "score__x__b", "score__y__value"]
    first = dict(zip(header, data[0], strict=True))
    assert first["sample_id"] == "q1" and first["output_tokens"] == "42"
    assert (
        first["score__x__a"] == "" and first["score__y__value"] == ""
    )  # missing and None both blank
    second = dict(zip(header, data[1], strict=True))
    assert second["output_tokens"] == "" and second["score__y__value"] == "6.0"


def test_empty_input_writes_header_only(tmp_path):
    assert write_jsonl([], tmp_path / "e.jsonl") == 0
    assert (tmp_path / "e.jsonl").read_text() == ""
    assert write_csv([], tmp_path / "e.csv") == 0
    assert (tmp_path / "e.csv").read_text().strip() == ",".join(FIXED_COLUMNS)
