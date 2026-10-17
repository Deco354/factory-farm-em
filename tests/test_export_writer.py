"""JSONL and CSV writers: row counts, null handling, and column order; which logs a
directory's export reads."""

import csv
import json

from inspect_ai.log import EvalConfig, EvalDataset, EvalLog, EvalSample, EvalSpec, write_eval_log

from fragile_compassion.export.schema import FIXED_COLUMNS
from fragile_compassion.export.writer import iter_rows, superseded, write_csv, write_jsonl


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


# ---- logs an eval_set retry leaves behind


def test_superseded_keeps_the_latest_log_of_each_task():
    logs = [
        ("a-crashed.eval", "task-a", "2026-10-09T13:20:16+00:00"),
        ("a-retry.eval", "task-a", "2026-10-09T13:46:48+00:00"),
        ("b.eval", "task-b", "2026-10-09T13:20:16+00:00"),
    ]
    assert superseded(logs) == {"a-crashed.eval": "a-retry.eval"}
    assert superseded(iter(logs)) == {"a-crashed.eval": "a-retry.eval"}  # one-pass input
    assert superseded(logs[::-1]) == {"a-crashed.eval": "a-retry.eval"}  # order-free
    # Compared as times, not text: the same instant in another offset is not "later".
    tz = [
        ("x.eval", "t", "2026-10-09T14:00:00+01:00"),
        ("y.eval", "t", "2026-10-09T13:30:00+00:00"),
    ]
    assert superseded(tz) == {"x.eval": "y.eval"}


def write_log(path, task_id, created, status, sample_ids):
    log = EvalLog(
        eval=EvalSpec(
            created=created,
            task="plumbing_task",
            task_id=task_id,
            dataset=EvalDataset(samples=len(sample_ids), sample_ids=list(sample_ids)),
            model="mockllm/model",
            config=EvalConfig(),
        ),
        status=status,
        samples=[EvalSample(id=s, epoch=1, input="q", target="") for s in sample_ids],
    )
    write_eval_log(log, str(path))


def test_export_skips_a_crashed_attempt_log_its_retry_replaces(tmp_path):
    # hb-em-001 (2026-10-09): the crashed attempt's `started` log held finished samples
    # that the retry log also holds; reading both counted them twice.
    old = tmp_path / "2026-10-09T13-20-16_a.eval"
    write_log(old, "task-a", "2026-10-09T13:20:16+00:00", "started", ["s1"])
    new = tmp_path / "2026-10-09T13-46-48_a.eval"
    write_log(new, "task-a", "2026-10-09T13:46:48+00:00", "success", ["s1", "s2"])
    other = tmp_path / "2026-10-09T13-20-16_b.eval"
    write_log(other, "task-b", "2026-10-09T13:20:16+00:00", "success", ["s1"])
    skipped = {}
    rows = list(iter_rows(tmp_path, skipped=skipped))
    assert sorted((r["task_id"], r["sample_id"], r["status"]) for r in rows) == [
        ("task-a", "s1", "success"),
        ("task-a", "s2", "success"),
        ("task-b", "s1", "success"),
    ]
    assert [(k.rsplit("/", 1)[-1], v.rsplit("/", 1)[-1]) for k, v in skipped.items()] == [
        (old.name, new.name)
    ]
