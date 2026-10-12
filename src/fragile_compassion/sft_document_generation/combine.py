"""Combines the passing pairs of several runs into one training set.

Each run directory's master_dataset.json is read; records are merged by ID (the same ID
with different content is an error); only pairs that pass the same rule as per-run
consolidation (generate.pair_passes) are kept. One train_<arm>.jsonl is written per arm,
with pairs in the same shuffled order in every file, so line N of each arm file is the
same pair. A manifest records which runs went in and how the pairs break down.

Usage, from the repo root:

    uv run python -m fragile_compassion.sft_document_generation.combine \\
        output/all-03-broiler output/all-03-breeder output/all-03-swine --name pilot-01

The result goes to output/combined-<name>/ (gitignored, like all generated data).
"""

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from fragile_compassion.sft_document_generation import generate as g

# ---------------------------------------------------------------------------
# Pure functions
# ---------------------------------------------------------------------------


def merge_records(runs: list[tuple[str, list[dict]]]) -> dict[str, dict]:
    """Merges (run name, records) lists by record ID.

    The same record read twice (for example, a run listed twice) is kept once. Two
    different records with the same ID raise ValueError: IDs are built from card,
    persona, task and prompt, so a clash means the inputs are inconsistent.
    """
    merged: dict[str, dict] = {}
    seen_in: dict[str, str] = {}
    clashes = []
    for run_name, records in runs:
        for r in records:
            rid = r.get("id")
            if not rid:
                continue
            if rid in merged and merged[rid] != r:
                clashes.append(f"{rid} ({seen_in[rid]} vs {run_name})")
                continue
            merged.setdefault(rid, r)
            seen_in.setdefault(rid, run_name)
    if clashes:
        raise ValueError("records with the same ID but different content: " + ", ".join(clashes))
    return merged


def shuffled_pair_ids(pair_ids, seed: int) -> list[str]:
    """Pair IDs in a reproducible shuffled order (sorted first, so input order is irrelevant)."""
    ordered = sorted(pair_ids)
    random.Random(seed).shuffle(ordered)
    return ordered


def training_files(passing: dict[str, list[dict]], order: list[str]) -> dict[str, list[dict]]:
    """{arm: training rows}, every arm following the same pair order."""
    files: dict[str, list[dict]] = {}
    for pair_id in order:
        for r in passing[pair_id]:
            files.setdefault(r["arm"], []).append(g.training_row(r))
    return files


def count_by(pairs: list[list[dict]], key) -> dict[str, int]:
    """Pairs counted by key(first record of the pair), most common first."""
    counts = Counter(str(key(pair[0])) for pair in pairs)
    return dict(counts.most_common())


def build_manifest(
    run_names: list[str],
    records: dict[str, dict],
    by_pair: dict[str, list[dict]],
    passing: dict[str, list[dict]],
    files: dict[str, list[dict]],
    seed: int,
) -> dict:
    """What went into the combined set and how the kept pairs break down."""
    kept = list(passing.values())
    versions = sorted({str(r.get("prompt_version")) for r in records.values()})
    models = sorted({str(r.get("teacher_model")) for r in records.values()})
    warnings = []
    if len(versions) > 1:
        warnings.append(f"records come from more than one prompt version: {versions}")
    if len(models) > 1:
        warnings.append(f"records come from more than one teacher model: {models}")
    return {
        "runs": run_names,
        "shuffle_seed": seed,
        "records_read": len(records),
        "pairs_read": len(by_pair),
        "pairs_kept": len(kept),
        "rows_per_arm": {arm: len(rows) for arm, rows in sorted(files.items())},
        "prompt_versions": versions,
        "teacher_models": models,
        "pairs_by_source": count_by(kept, lambda r: r.get("source", {}).get("title")),
        "pairs_by_card": count_by(kept, lambda r: r.get("source", {}).get("card_id")),
        "pairs_by_task": count_by(kept, lambda r: r.get("task")),
        "pairs_by_persona": count_by(kept, lambda r: r.get("persona", {}).get("persona_id")),
        "pairs_by_persona_seed": count_by(kept, lambda r: r.get("persona_seed")),
        "pairs_by_run": count_by(kept, lambda r: r.get("run_id")),
        "warnings": warnings,
    }


def combine(runs: list[tuple[str, list[dict]]], seed: int = 0) -> tuple[dict, dict, dict]:
    """(merged records, training files by arm, manifest) for the given runs. Pure."""
    records = merge_records(runs)
    by_pair = g.group_pairs(records.values())
    passing = g.passing_pairs(by_pair)
    files = training_files(passing, shuffled_pair_ids(passing, seed))
    manifest = build_manifest([name for name, _ in runs], records, by_pair, passing, files, seed)
    return records, files, manifest


# ---------------------------------------------------------------------------
# Files and CLI
# ---------------------------------------------------------------------------


def load_run(run_dir: Path) -> tuple[str, list[dict]]:
    path = Path(run_dir) / "master_dataset.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    return Path(run_dir).name, data.get("records", [])


def write_combined(out_dir: Path, records: dict, files: dict, manifest: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("train_*.jsonl"):
        old.unlink()
    master = {"records": list(records.values())}
    (out_dir / "master_dataset.json").write_text(
        json.dumps(master, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    for arm, rows in files.items():
        lines = "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n"
        (out_dir / f"train_{arm}.jsonl").write_text(lines, encoding="utf-8")
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def main():
    parser = argparse.ArgumentParser(description="Combine passing pairs from several runs.")
    parser.add_argument("run_dirs", nargs="+", type=Path, help="Run directories to combine.")
    parser.add_argument("--name", required=True, help="Written to output/combined-<name>/.")
    parser.add_argument("--seed", type=int, default=0, help="Shuffle seed for pair order.")
    parser.add_argument("--force", action="store_true", help="Overwrite an existing output.")
    args = parser.parse_args()

    out_dir = g.ROOT_DIR / "output" / f"combined-{args.name}"
    if out_dir.exists() and not args.force:
        raise SystemExit(f"{out_dir} already exists; pass --force to overwrite it.")

    runs = [load_run(d) for d in args.run_dirs]
    records, files, manifest = combine(runs, args.seed)
    write_combined(out_dir, records, files, manifest)

    print(f"Combined {len(runs)} run(s) into {out_dir}")
    print(f"Pairs kept: {manifest['pairs_kept']} of {manifest['pairs_read']}")
    for arm, n in manifest["rows_per_arm"].items():
        print(f"  train_{arm}.jsonl: {n} rows")
    for label in ("pairs_by_source", "pairs_by_card", "pairs_by_task"):
        print(f"{label}: {manifest[label]}")
    for w in manifest["warnings"]:
        print(f"WARNING: {w}")


if __name__ == "__main__":
    main()
