"""Walk a log directory and write per-item rows as JSONL (and optionally CSV)."""

from __future__ import annotations

import csv
import json
import sys
from collections.abc import Iterable, Iterator
from datetime import datetime
from pathlib import Path
from typing import Any

from inspect_ai.log import list_eval_logs, read_eval_log, read_eval_log_samples

from fragile_compassion.export.rows import header_from_log, sample_to_rows
from fragile_compassion.export.schema import FIXED_COLUMNS


def superseded(logs: Iterable[tuple[str, str, str]]) -> dict[str, str]:
    """{location: replacing location} for every log that a later log of the same task
    replaces, from (location, task_id, created) triples.

    An eval_set retry after a crash writes a new log for each unfinished task, with the
    same task id and a later `eval.created`, and leaves the crashed attempt's log beside
    it (hb-em-001, 2026-10-09). Exporting both would count those samples twice, and the
    retry log already holds the samples the crashed attempt finished."""
    logs = list(logs)
    latest: dict[str, tuple[datetime, str]] = {}
    for location, task_id, created in logs:
        key = (datetime.fromisoformat(created), location)
        if task_id not in latest or key > latest[task_id]:
            latest[task_id] = key
    return {
        location: latest[task_id][1]
        for location, task_id, _created in logs
        if location != latest[task_id][1]
    }


def iter_rows(
    log_dir: str | Path, *, with_text: bool = False, skipped: dict[str, str] | None = None
) -> Iterator[dict[str, Any]]:
    """Rows from every log under log_dir except superseded ones, which go in `skipped`."""
    logs = [(info, read_eval_log(info, header_only=True)) for info in list_eval_logs(str(log_dir))]
    replaced = superseded([(info.name, log.eval.task_id, log.eval.created) for info, log in logs])
    if skipped is not None:
        skipped.update(replaced)
    for info, log in logs:
        if info.name in replaced:
            continue
        header = header_from_log(log)
        for sample in read_eval_log_samples(info, all_samples_required=False):
            yield from sample_to_rows(header, sample, with_text=with_text)


def write_jsonl(rows: Iterable[dict[str, Any]], path: str | Path) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, default=str))
            f.write("\n")
            n += 1
    return n


def write_csv(rows: list[dict[str, Any]], path: str | Path) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    extra = sorted({k for r in rows for k in r} - set(FIXED_COLUMNS))
    columns = [*FIXED_COLUMNS, *extra]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if v is None else v) for k, v in r.items()})
    return len(rows)


def export(
    log_dir: str | Path, out: str | Path, *, csv_too: bool = False, with_text: bool = False
) -> int:
    skipped: dict[str, str] = {}
    rows = list(iter_rows(log_dir, with_text=with_text, skipped=skipped))
    for old, new in sorted(skipped.items()):
        print(f"skipped {old}: superseded by {new} (same task)", file=sys.stderr)
    n = write_jsonl(rows, out)
    if csv_too:
        write_csv(rows, Path(out).with_suffix(".csv"))
    return n
