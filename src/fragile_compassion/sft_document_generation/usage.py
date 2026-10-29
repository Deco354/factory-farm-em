"""Token and cost accounting for API calls.

OpenRouter returns the tokens and the cost (in credits, i.e. US dollars) of every request in
the response's `usage` object. generate.request_with_retries records one row per API call,
including calls that are later retried or fail, since those are billed too.

Rows are appended to <run dir>/usage.jsonl as each call returns, so an interrupted run still
has its cost on disk. At the end of a run, finish_log writes <run dir>/usage_summary.json,
prints the totals and appends one line per run to output/usage_ledger.jsonl.

To total one or more runs afterwards (including interrupted ones), from the repo root:

    uv run python -m fragile_compassion.sft_document_generation.usage output/all-05-*

Each row keeps OpenRouter's generation ID, which can be looked up in the OpenRouter
dashboard or at /api/v1/generation?id=<id>.
"""

import argparse
import json
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
from typing import Any

USAGE_FILE = "usage.jsonl"
SUMMARY_FILE = "usage_summary.json"
LEDGER_FILE = "usage_ledger.jsonl"
SUM_KEYS = ("prompt_tokens", "completion_tokens", "reasoning_tokens", "cached_tokens", "cost")

# ---------------------------------------------------------------------------
# Pure functions
# ---------------------------------------------------------------------------


def usage_row(
    usage: dict | None,
    model: str,
    generation_id: str | None = None,
    finish_reason: str | None = None,
    context: dict | None = None,
) -> dict:
    """One call's tokens and cost from an OpenRouter `usage` object. Missing fields are 0."""
    u = usage or {}
    prompt_details = u.get("prompt_tokens_details") or {}
    completion_details = u.get("completion_tokens_details") or {}
    return {
        **(context or {}),
        "model": model,
        "generation_id": generation_id,
        "finish_reason": finish_reason,
        "prompt_tokens": int(u.get("prompt_tokens") or 0),
        "completion_tokens": int(u.get("completion_tokens") or 0),
        "reasoning_tokens": int(completion_details.get("reasoning_tokens") or 0),
        "cached_tokens": int(prompt_details.get("cached_tokens") or 0),
        "cost": float(u.get("cost") or 0.0),
    }


def totals(rows: Iterable[dict]) -> dict:
    """Calls, token counts and cost summed over rows. `calls_without_cost` counts rows
    where the provider reported no cost, so a total of $0 can't be mistaken for free."""
    out: dict[str, Any] = {"calls": 0, **{k: 0 for k in SUM_KEYS}, "calls_without_cost": 0}
    for row in rows:
        out["calls"] += 1
        for key in SUM_KEYS:
            out[key] += row.get(key) or 0
        out["calls_without_cost"] += not row.get("cost")
    out["cost"] = round(out["cost"], 6)
    return out


def totals_by(rows: list[dict], key: str) -> dict[str, dict]:
    """totals() for each value of `key` (e.g. model or card_id), most expensive first."""
    groups: dict[str, list[dict]] = {}
    for row in rows:
        groups.setdefault(str(row.get(key)), []).append(row)
    summed = {k: totals(v) for k, v in groups.items()}
    return dict(sorted(summed.items(), key=lambda kv: -kv[1]["cost"]))


def format_totals(t: dict) -> str:
    """One line for the console."""
    line = (
        f"{t['calls']} calls, {t['prompt_tokens']:,} prompt + {t['completion_tokens']:,} "
        f"completion tokens ({t['reasoning_tokens']:,} reasoning), ${t['cost']:.4f}"
    )
    if t["calls_without_cost"]:
        line += f" ({t['calls_without_cost']} call(s) reported no cost)"
    return line


def summary(rows: list[dict], run_id: str, script: str) -> dict:
    """What usage_summary.json holds."""
    by_key = next((k for k in ("card_id", "chunk") if any(k in r for r in rows)), None)
    out = {
        "run_id": run_id,
        "script": script,
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "totals": totals(rows),
        "by_model": totals_by(rows, "model"),
    }
    if by_key:
        out[f"by_{by_key}"] = totals_by(rows, by_key)
    return out


# ---------------------------------------------------------------------------
# Logging during a run
# ---------------------------------------------------------------------------

_log_path: Path | None = None
_context: dict[str, Any] = {}


def start_log(run_dir: Path) -> Path:
    """Sends every later call's usage row to <run_dir>/usage.jsonl (appending, so a
    rerun into the same folder keeps the earlier calls)."""
    global _log_path
    _log_path = Path(run_dir) / USAGE_FILE
    _log_path.parent.mkdir(parents=True, exist_ok=True)
    return _log_path


def set_context(**fields: Any) -> None:
    """Labels added to every later row until changed, e.g. card_id and persona_seed."""
    _context.clear()
    _context.update(fields)


def as_dict(usage: Any) -> dict:
    """The SDK's usage object as a dict, keeping OpenRouter's extra fields (cost)."""
    if usage is None:
        return {}
    if hasattr(usage, "model_dump"):
        return usage.model_dump()
    return dict(usage)


def record(response: Any, model: str) -> dict:
    """Builds the row for one API response, appends it to the log (if started) and
    returns it."""
    choice = (getattr(response, "choices", None) or [None])[0]
    row = usage_row(
        as_dict(getattr(response, "usage", None)),
        model,
        getattr(response, "id", None),
        getattr(choice, "finish_reason", None),
        {"time": datetime.now().isoformat(timespec="seconds"), **_context},
    )
    if _log_path is not None:
        with _log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return row


def read_rows(run_dir: Path) -> list[dict]:
    path = Path(run_dir) / USAGE_FILE
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def finish_log(run_dir: Path, run_id: str, script: str, ledger_dir: Path) -> dict:
    """Writes usage_summary.json, appends the run to the ledger and prints the totals."""
    rows = read_rows(run_dir)
    result = summary(rows, run_id, script)
    (Path(run_dir) / SUMMARY_FILE).write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    ledger_dir.mkdir(parents=True, exist_ok=True)
    line = {"run_id": run_id, "script": script, "finished_at": result["finished_at"]}
    with (ledger_dir / LEDGER_FILE).open("a", encoding="utf-8") as f:
        f.write(json.dumps({**line, **result["totals"]}) + "\n")
    print(f"Usage: {format_totals(result['totals'])}")
    print(f"Usage details: {Path(run_dir) / SUMMARY_FILE}")
    return result


# ---------------------------------------------------------------------------
# CLI: total one or more run folders
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Total the API usage and cost of runs.")
    parser.add_argument("run_dirs", nargs="+", type=Path, help="Run folders with usage.jsonl.")
    args = parser.parse_args()
    run_dirs = []
    for arg in args.run_dirs:  # PowerShell passes wildcards through unexpanded
        run_dirs += sorted(Path().glob(str(arg))) if "*" in str(arg) else [arg]
    everything = []
    for run_dir in run_dirs:
        rows = read_rows(run_dir)
        if not rows:
            print(f"{run_dir}: no {USAGE_FILE}")
            continue
        everything += rows
        print(f"{run_dir}: {format_totals(totals(rows))}")
    if len(run_dirs) > 1 and everything:
        print(f"All runs: {format_totals(totals(everything))}")


if __name__ == "__main__":
    main()
