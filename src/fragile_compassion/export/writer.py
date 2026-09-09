"""Walk a log directory and write per-item rows as JSONL (and optionally CSV)."""

from __future__ import annotations

import csv
import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from inspect_ai.log import list_eval_logs, read_eval_log, read_eval_log_samples

from fragile_compassion.export.rows import header_from_log, sample_to_rows
from fragile_compassion.export.schema import FIXED_COLUMNS


def iter_rows(log_dir: str | Path, *, with_text: bool = False) -> Iterator[dict[str, Any]]:
    for info in list_eval_logs(str(log_dir)):
        header = header_from_log(read_eval_log(info, header_only=True))
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
    rows = list(iter_rows(log_dir, with_text=with_text))
    n = write_jsonl(rows, out)
    if csv_too:
        write_csv(rows, Path(out).with_suffix(".csv"))
    return n
