"""Draft procedure cards from book excerpts with the teacher model, for human review.

Each job in configs/procedure_card_jobs.yaml names a cleaned book, line ranges, the purposes
and domain/sector values allowed, and the procedures to draft. One model call per job returns
cards; code then fills fixed fields (stakes, outcome, reviewed_by: null), checks every card
with check_cards (quotes against the book, required fields) and writes the drafts to
output/procedures/card-drafts/<run_id>/<job>.yaml. Problems are recorded on each card under
`draft_issues`. Nothing is written to configs/procedure_cards/: review the drafts, fix or
delete cards, remove `draft_issues`, and copy the cards into a card file there.

    uv run python -m fragile_compassion.procedure_generation.draft_procedure_cards \
        --run-id drafts-01
    uv run python -m fragile_compassion.procedure_generation.draft_procedure_cards \
        --job msu_disbudding_calves --run-id drafts-01
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import yaml

from fragile_compassion.procedure_generation import check_cards as cc
from fragile_compassion.procedure_generation import generate_procedures as gp
from fragile_compassion.sft_document_generation import generate as g
from fragile_compassion.sft_document_generation import generate_cards as gc
from fragile_compassion.sft_document_generation import usage

JOBS_PATH = Path("configs/procedure_card_jobs.yaml")
PROMPT_PATH = Path("configs/prompts/procedures/procedure_card_drafter.md")
DRAFT_DIR = Path("output/procedures/card-drafts")
SOURCE_KEYS = ("title", "edition", "mode", "outcome", "jurisdiction", "book_text")
CARD_KEYS = (
    "card_id",
    "procedure",
    "species",
    "domain",
    "sector",
    "method",
    "scenarios",
    "steps",
    "checks",
    "attentive_points",
)


def load_jobs(path: Path) -> list[dict]:
    return (yaml.safe_load(path.read_text(encoding="utf-8")) or {}).get("jobs", [])


def excerpt(book_lines: list[str], ranges: list[list[int]]) -> str:
    """The given 1-based inclusive line ranges, joined with a gap marker. Pure."""
    parts = ["\n".join(book_lines[a - 1 : b]) for a, b in ranges]
    return "\n\n[...]\n\n".join(parts)


def build_payload(job: dict, text: str, existing_ids: set[str]) -> str:
    src = job["source"]
    allowed = "; ".join(f"domain {d}, sector {s}" for d, s in job["allowed"])
    return "\n\n".join(
        [
            f"SOURCE: {src['title']} ({src.get('edition', '')}); outcome {src['outcome']}; "
            f"jurisdiction {src['jurisdiction']}",
            f"PURPOSES: {', '.join(job['purposes'])}",
            f"ALLOWED: {allowed}",
            f"PROCEDURES: {job['procedures']}",
            f"MAX_CARDS: {job.get('max_cards', 3)}",
            f"ID_PREFIX: {job['id_prefix']}",
            f"EXISTING_IDS: {', '.join(sorted(existing_ids)) or 'none'}",
            f'EXCERPT:\n"""\n{text}\n"""',
        ]
    )


def finalize(raw: dict, job: dict, book_norm: str, taken: set[str]) -> dict:
    """A draft card in the card-file layout, with draft_issues for anything to fix. Pure
    apart from adding its ID to `taken`."""
    card = {k: raw.get(k) for k in CARD_KEYS}
    card["stakes"] = "high"
    card["reviewed_by"] = None
    issues = []
    allowed = {tuple(x) for x in job["allowed"]}
    if (card.get("domain"), card.get("sector")) not in allowed:
        issues.append(
            f"domain/sector {card.get('domain')}/{card.get('sector')} not in {sorted(allowed)}"
        )
    extra = sorted(set(card.get("scenarios") or {}) - set(job["purposes"]))
    if extra:
        issues.append(f"purposes not allowed for this job: {extra}")
    cid = card.get("card_id") or ""
    if not cid.startswith(job["id_prefix"]):
        issues.append(f"card_id should start with {job['id_prefix']}")
    if cid in taken:
        issues.append("card_id already used")
    taken.add(cid)
    resolved = cc.resolved(card, {"outcome": job["source"]["outcome"]})
    issues += cc.check_card(resolved, book_norm)
    if issues:
        card["draft_issues"] = issues
    return card


def existing_card_ids(card_dir: Path) -> set[str]:
    ids = set()
    for path in card_dir.glob("*.yaml"):
        ids |= {c.get("card_id") for c in (cc.load(path).get("cards") or [])}
    return ids


def write_drafts(path: Path, job: dict, cards: list[dict]) -> None:
    head = (
        f"# DRAFT procedure cards from job {job['name']} ({datetime.now():%Y-%m-%d}).\n"
        "# Review every card against the book. Fix or delete cards listed under draft_issues,\n"
        "# then remove draft_issues and copy the cards into a file in configs/procedure_cards/.\n\n"
    )
    body = {"source": {k: job["source"][k] for k in SOURCE_KEYS}, "cards": cards}
    path.write_text(
        head + yaml.safe_dump(body, sort_keys=False, allow_unicode=True, width=100),
        encoding="utf-8",
    )


def run_job(job, settings, client, root, run_dir, taken, dry_run) -> str:
    book_path = root / job["source"]["book_text"]
    lines = book_path.read_text(encoding="utf-8").splitlines()
    payload = build_payload(job, excerpt(lines, job["lines"]), taken)
    if dry_run:
        (run_dir / f"{job['name']}.payload.txt").write_text(payload, encoding="utf-8")
        return f"payload {len(payload):,} chars (dry run)"
    usage.set_context(card_id=job["name"])
    parsed, _ = gc.call_json(
        client,
        settings["model"],
        settings["prompt"],
        payload,
        settings["temperature"],
        settings["max_tokens"],
        settings["reasoning"],
        truncation_retries=2,
    )
    book_norm = cc.norm_ws("\n".join(lines))
    cards = [finalize(c, job, book_norm, taken) for c in parsed.get("cards") or []]
    write_drafts(run_dir / f"{job['name']}.yaml", job, cards)
    flagged = sum(1 for c in cards if c.get("draft_issues"))
    return f"{len(cards)} cards, {flagged} with draft_issues"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Draft procedure cards from book excerpts.")
    ap.add_argument("--config", type=Path, default=g.DEFAULT_CONFIG_PATH)
    ap.add_argument("--jobs", type=Path, default=JOBS_PATH)
    ap.add_argument("--job", action="append", help="Only this job name (repeatable).")
    ap.add_argument("--run-id", help="Defaults to a timestamp.")
    ap.add_argument("--workers", type=int, default=4, help="Jobs to run at once.")
    ap.add_argument("--dry-run", action="store_true", help="Write payloads; no API calls.")
    args = ap.parse_args(argv)

    root = g.ROOT_DIR
    jobs = [j for j in load_jobs(root / args.jobs) if not args.job or j["name"] in args.job]
    if not jobs:
        print("No jobs selected.")
        return 1
    cfg = g.load_config(args.config)
    d = cfg.get("procedure_drafting") or cfg.get("drafting") or {}
    settings = {
        "model": d.get("teacher_model") or cfg.get("api", {}).get("teacher_model"),
        "temperature": float(d.get("temperature", 0.2)),
        "max_tokens": int(d.get("max_tokens", 32000)),
        "reasoning": d.get("reasoning", {"effort": "medium"}),
        "prompt": (root / PROMPT_PATH).read_text(encoding="utf-8"),
    }
    run_id = args.run_id or datetime.now().strftime("drafts%Y%m%d_%H%M%S")
    run_dir = root / DRAFT_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    client = None if args.dry_run else g.make_client(cfg)
    usage.start_log(run_dir)
    taken = existing_card_ids(root / "configs/procedure_cards")
    print(f"Run {run_id}: {len(jobs)} job(s) -> {run_dir}")

    def one(job):
        try:
            msg = run_job(job, settings, client, root, run_dir, taken, args.dry_run)
        except Exception as e:  # one failed job should not stop the others
            msg = f"FAILED: {e.__class__.__name__}: {e}"
            (run_dir / f"{job['name']}.failed.json").write_text(
                json.dumps({"job": job["name"], "error": str(e)}, indent=2), encoding="utf-8"
            )
        gp.say(job["name"], msg)
        return msg

    with gp.thread_safe_usage(), ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        list(pool.map(one, jobs))
    if not args.dry_run:
        usage.finish_log(run_dir, run_id, "draft_procedure_cards", root / gp.OUTPUT_DIR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
