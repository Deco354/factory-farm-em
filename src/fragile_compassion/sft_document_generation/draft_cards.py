"""Drafts parameter cards from a book's ranked chunks, for a person to review.

The model reads one chunk at a time and proposes cards (configs/prompts/card_drafter.md, or
card_drafter_crop.md when the source's mode is crop).
Code then repairs near-miss quotes, drops duplicates, and runs the card checker on every
draft. The result is a card file under output/card-drafts/<run-id>/ with passing drafts
first and failing drafts after, each with the checker's issues attached. Nothing here is
generated from: drafts have no reviewed_by, and generate_cards skips unreviewed cards.

Reviewing: for each draft, read the chunk it came from (draft.chunk), fix it in the
drafts file if needed, and set `decision: keep | park | skip` (park also needs `why:`).
promote_drafts then moves the decided drafts into the real card file in configs/cards/.

Usage, from the repo root (after process_book_ranked has written the chunks):

    uv run python -m fragile_compassion.sft_document_generation.draft_cards \\
        --book cobb-broiler --top 100 --run-id cobb-broiler-comp-01

`--book` looks up the card file, chunk folder, companion file and ID prefix in
configs/books.yaml. Any of them can still be given directly (--source, --chunks-dir,
--companion, --id-prefix), which overrides the registry; --no-companion sends none.
`--source` is any YAML file with a `source:` block (a card file, or a new file holding
only the block).

Decisions are not drafted again when they match (same category, unit and range, or the
same two options) any of:
  - a live or parked card in the source file or in an --existing file;
  - a passing draft from an earlier run of the same book under output/card-drafts/
    (found by source title; folders starting with "_" are skipped). Failing drafts are
    not counted, so a companion can fix them. --no-history turns this off.
Each run lists what it dropped under `duplicates_dropped`.
"""

import argparse
import difflib
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from fragile_compassion.sft_document_generation import books as bk
from fragile_compassion.sft_document_generation import cards as cd
from fragile_compassion.sft_document_generation import generate as g
from fragile_compassion.sft_document_generation import generate_cards as gc
from fragile_compassion.sft_document_generation import usage

CARD_KEYS = [
    "card_id",
    "parameter",
    "parameter_category",
    "unit",
    "range_low",
    "range_high",
    "range_quote",
    "output_end",
    "output_reason_basis",
    "output_end_reason",
    "output_end_quote",
    "other_reason_basis",
    "other_end_reason",
    "other_end_quote",
    "management_conditions",
    "production_consequences",
    "page",
    "notes",
    "reviewed_by",
]
QUOTE_FIELDS = ("range_quote", "output_end_quote", "other_end_quote", "legal_quote")
CJK = re.compile(r"[\u3040-\u30ff\u3400-\u9fff\uac00-\ud7af]")
REPAIR_MIN_RATIO = 0.92  # how close a near-miss quote must be to a book passage

# ---------------------------------------------------------------------------
# Pure functions
# ---------------------------------------------------------------------------


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", str(text).lower()).strip("-")


def tidy(value):
    """Whole floats as ints (28.0 -> 28), other values unchanged."""
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def normalize_draft(raw: dict, id_prefix: str, chunk_name: str) -> dict:
    """A model draft in card-file shape: known keys in order, defaults filled in."""
    keys = list(CARD_KEYS)
    if raw.get("deprivation"):  # only deprivation_duration cards carry this field
        keys.insert(keys.index("unit") + 1, "deprivation")
    if raw.get("stakes"):  # only high-stakes cards carry this field
        keys.insert(keys.index("card_id") + 1, "stakes")
    legal = [k for k in ("legal_end", "legal_quote", "legal_wording") if raw.get(k)]
    if legal:  # only legal-limit cards carry these
        at = keys.index("output_end") + 1
        keys[at:at] = legal
    if raw.get("parameter_category") == g.CHOICE_CATEGORY:  # two options, no unit or range
        at = keys.index("unit")
        keys[at : at + 3] = ["option_low", "option_high"]
    card: dict[str, Any] = {key: raw.get(key) for key in keys}
    base = slug(raw.get("card_id") or raw.get("parameter") or "card")
    prefix = slug(id_prefix)
    card["card_id"] = base if not prefix or base.startswith(prefix + "-") else f"{prefix}-{base}"
    for key in ("range_low", "range_high"):
        if key in card:
            card[key] = tidy(card[key])
    for key in ("management_conditions", "production_consequences"):
        card[key] = [e for e in card[key] or [] if isinstance(e, dict)]
    card["reviewed_by"] = None  # only a person sets this
    card["draft"] = {"chunk": chunk_name}
    return card


def drafts_from_reply(parsed: dict, id_prefix: str, chunk_name: str) -> tuple[list, list]:
    """(normalized cards, rejected candidates) from one drafter reply."""
    cards = [
        normalize_draft(c, id_prefix, chunk_name)
        for c in parsed.get("cards") or []
        if isinstance(c, dict)
    ]
    rejected = [
        {**r, "chunk": chunk_name} for r in parsed.get("rejected") or [] if isinstance(r, dict)
    ]
    return cards, rejected


def clean_text(text: str) -> str:
    """Whitespace-normalized text without stray ``` markers, as the checker sees it."""
    return g.norm_ws((text or "").replace("```", " "))


def repair_quote(quote: str, passage: str) -> str | None:
    """The passage text a near-miss quote was meant to copy, or None.

    Slides windows of about the quote's length over `passage` and takes the closest.
    Accepts it only if it is at least REPAIR_MIN_RATIO similar and contains exactly the
    same numbers, so a repair can fix punctuation, spacing or a dropped word, never a
    value.
    """
    target = clean_text(quote)
    words = clean_text(passage).split()
    n = len(target.split())
    if not n or not words:
        return None
    best, best_ratio = None, 0.0
    matcher = difflib.SequenceMatcher(autojunk=False)
    matcher.set_seq2(target)
    for size in range(max(1, n - 3), n + 4):
        for start in range(0, max(1, len(words) - size + 1)):
            candidate = " ".join(words[start : start + size])
            matcher.set_seq1(candidate)
            if matcher.real_quick_ratio() < best_ratio or matcher.quick_ratio() < best_ratio:
                continue
            ratio = matcher.ratio()
            if ratio > best_ratio:
                best, best_ratio = candidate, ratio
    if best is None or best_ratio < REPAIR_MIN_RATIO:
        return None
    if g.numbers_in(best, words=True) != g.numbers_in(target, words=True):
        return None
    return best


def repair_card_quotes(card: dict, passage: str, book_text: str) -> list[dict]:
    """Repairs text quotes not found in the book, in place. Returns what was changed."""
    book_n = clean_text(book_text)
    repairs = []

    def fix(quote):
        if not isinstance(quote, str) or clean_text(quote) in book_n:
            return quote
        repaired = repair_quote(quote, passage)
        if repaired and repaired in book_n:
            repairs.append({"from": quote, "to": repaired})
            return repaired
        return quote

    for key in QUOTE_FIELDS:
        value = card.get(key)
        if isinstance(value, list):
            card[key] = [fix(q) for q in value]
        else:
            card[key] = fix(value)
    for key in ("management_conditions", "production_consequences"):
        for entry in card.get(key) or []:
            entry["quote"] = fix(entry.get("quote"))
    return repairs


def draft_key(card: dict) -> tuple:
    """Two cards with the same unit, range and category are the same decision.
    Two choice cards with the same pair of options are the same decision."""
    if g.is_choice(card):
        options = sorted(g.norm_ws(g.option_label(card, e)).lower() for e in ("low", "high"))
        return (g.CHOICE_CATEGORY, *options)
    unit = re.sub(r"\s+", "", str(card.get("unit") or "").lower())
    unit = unit.replace("²", "2").replace("³", "3")
    try:
        low, high = g.canon_num(card.get("range_low")), g.canon_num(card.get("range_high"))
    except (TypeError, ValueError):
        low, high = str(card.get("range_low")), str(card.get("range_high"))
    return (card.get("parameter_category"), unit, low, high)


def dedupe(drafts: list[dict], existing: list[dict]) -> tuple[list[dict], list[dict]]:
    """(kept drafts, dropped duplicates). Drops drafts matching an existing card or an
    earlier draft; the earlier chunk's draft wins. Card IDs are made unique."""
    seen = {draft_key(c) for c in existing}
    ids = {c.get("card_id") for c in existing}
    kept, dropped = [], []
    for card in drafts:
        key = draft_key(card)
        if key in seen:
            dropped.append(card)
            continue
        seen.add(key)
        base, n = card["card_id"], 2
        while card["card_id"] in ids:
            card["card_id"], n = f"{base}-{n}", n + 1
        ids.add(card["card_id"])
        kept.append(card)
    return kept, dropped


SMALL_GAP = 0.15  # warn when the high end is less than 15% above the low end


def gap_warning(card: dict) -> list[str]:
    """A warning when the two ends are too close to be a real decision."""
    try:
        low, high = float(card["range_low"]), float(card["range_high"])
    except (KeyError, TypeError, ValueError):
        return []
    if low > 0 and (high - low) / low < SMALL_GAP:
        return [
            f"small gap: {g.canon_num(high)} is only {100 * (high - low) / low:.0f}% above "
            f"{g.canon_num(low)}; is this a real decision?"
        ]
    return []


def non_english(card: dict) -> bool:
    """True if any text in the draft is Chinese, Japanese or Korean (the drafter model
    sometimes answers in Chinese)."""
    import json

    return bool(CJK.search(json.dumps(card, ensure_ascii=False)))


def check_drafts(drafts: list[dict], book_text: str) -> None:
    """Attaches the checker's issues and the warnings to each draft, in place."""
    for card in drafts:
        issues = cd.check_card(card, book_text)
        if non_english(card):
            issues = ["draft is not in English"] + issues
        card["draft"]["issues"] = issues
        card["draft"]["warnings"] = gap_warning(card) + (
            [] if issues else cd.balance_warnings(card) + cd.quote_warnings(card)
        )


def drafts_file(source: dict, drafts: list[dict], rejected: list, duplicates: list) -> dict:
    """Card-file data: passing drafts first, then failing ones."""
    ordered = sorted(drafts, key=lambda c: bool(c["draft"].get("issues")))
    with_decision = [{"card_id": c["card_id"], "decision": c.get("decision"), **c} for c in ordered]
    return {
        "source": source,
        "cards": with_decision,
        "rejected_by_drafter": rejected,
        "duplicates_dropped": [
            {"card_id": c["card_id"], "parameter": c.get("parameter"), "chunk": c["draft"]["chunk"]}
            for c in duplicates
        ],
    }


def file_header(source_title: str, n_pass: int, n_fail: int) -> str:
    return (
        f"# DRAFT cards for {source_title}, written by draft_cards.py. Not reviewed.\n"
        f"# {n_pass} draft(s) pass the checker, {n_fail} fail (listed last, with issues).\n"
        "# For each draft: read its chunk, fix it here if needed, and set `decision:` to\n"
        "# keep, park (add a `why:`) or skip. Then run promote_drafts --run <this run>\n"
        "# --reviewer <initials>. A passing draft can still be wrong: the checker cannot\n"
        "# judge whether both ends are acceptable practice or whether the trade-off is real.\n\n"
    )


DRAFTER_PROMPTS = {
    "animal": ("prompt_path", "configs/prompts/card_drafter.md"),
    "crop": ("crop_prompt_path", "configs/prompts/card_drafter_crop.md"),
}


def drafter_prompt_path(source: dict, drafting_cfg: dict) -> str:
    """The drafter prompt for the source's mode: livestock and crop cards follow
    different rules (crop cards are the harmless control)."""
    mode = source.get("mode", "animal")
    if mode not in DRAFTER_PROMPTS:
        raise ValueError(f"source.mode must be one of {sorted(DRAFTER_PROMPTS)}, got {mode!r}")
    key, default = DRAFTER_PROMPTS[mode]
    return drafting_cfg.get(key, default)


def build_payload(source: dict, chunk_text: str, companions: list[str] | None = None) -> str:
    payload = (
        f"SOURCE: {source.get('title')} ({source.get('edition')}), {source.get('domain')}\n"
        f'EXCERPT:\n"""\n{chunk_text}\n"""'
    )
    for text in companions or []:
        payload += f'\nCOMPANION EXCERPT (same source):\n"""\n{text}\n"""'
    return payload


def chunks_containing(paths: list[Path], needles: list[str]) -> list[Path]:
    """The chunks whose text contains any of the needles (all chunks if none given)."""
    if not needles:
        return paths
    return [p for p in paths if any(n in g.load_file_content(p) for n in needles)]


# ---------------------------------------------------------------------------
# Files and CLI
# ---------------------------------------------------------------------------


def load_source(path: Path, include_parked: bool = False) -> tuple[dict, list[dict]]:
    """(source block, cards) from a YAML file whose `cards` may be missing. With
    include_parked, cards under `parked:` are returned too (for de-duplication only)."""
    import yaml

    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(data.get("source"), dict):
        raise ValueError(f"{path}: expected a top-level 'source' mapping.")
    keys = ("cards", "parked") if include_parked else ("cards",)
    found = [c for k in keys for c in data.get(k) or [] if isinstance(c, dict)]
    return data["source"], found


def history_drafts(
    drafts_root: Path, title: str, skip: set[str] = frozenset()
) -> tuple[list[dict], list[str]]:
    """(passing drafts, run folders) from earlier draft runs of the book with this title.

    Folders starting with "_" (e.g. _superseded) and those named in `skip` are ignored.
    Failing drafts are left out on purpose: a later run may fix them."""
    import yaml

    found, runs = [], []
    for path in sorted(Path(drafts_root).glob("*/*_drafts.yaml")):
        run = path.parent.name
        if run.startswith("_") or run in skip:
            continue
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError:
            continue
        if (data.get("source") or {}).get("title") != title:
            continue
        passing = [
            c
            for c in data.get("cards") or []
            if isinstance(c, dict) and not (c.get("draft") or {}).get("issues")
        ]
        if passing:
            found += passing
            runs.append(run)
    return found, runs


def apply_book(
    args: argparse.Namespace, books_path: Path | None = None, root: Path | None = None
) -> argparse.Namespace:
    """Fills --source, --chunks-dir, --companion and --id-prefix from the book registry.
    Values given on the command line win; --no-companion sends no companion."""
    if args.book:
        entry = bk.get_book(args.book, books_path or bk.BOOKS_PATH, root or g.ROOT_DIR)
        if entry["status"] != "active":
            print(f"Note: book {args.book!r} is marked {entry['status']} in the registry.")
        args.source = args.source or entry["card_file"]
        args.chunks_dir = args.chunks_dir or entry["chunks_dir"]
        args.id_prefix = args.id_prefix or entry["id_prefix"]
        if not args.companion and not args.no_companion:
            if Path(entry["companion"]).exists():
                args.companion = [entry["companion"]]
            else:
                print(f"Note: no companion file at {entry['companion']} (run make_companion).")
    if args.no_companion:
        args.companion = []
    if not args.source or not args.chunks_dir:
        raise SystemExit("Give --book, or both --source and --chunks-dir.")
    return args


def ranked_chunks(chunks_dir: Path, top: int) -> list[Path]:
    paths = [p for p in Path(chunks_dir).glob("*_rank*_pos*.txt")]
    paths.sort(key=lambda p: g.parse_chunk_filename(p).get("rank", 10**6))
    return paths[:top]


def write_drafts(out_path: Path, data: dict) -> None:
    import yaml

    n_fail = sum(bool(c["draft"].get("issues")) for c in data["cards"])
    body = yaml.safe_dump(data, sort_keys=False, allow_unicode=True, width=100)
    header = file_header(data["source"].get("title", "?"), len(data["cards"]) - n_fail, n_fail)
    out_path.write_text(header + body, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Draft parameter cards for review.")
    parser.add_argument("--config", type=Path, default=g.DEFAULT_CONFIG_PATH)
    parser.add_argument("--book", help="Book ID from configs/books.yaml (fills the paths below).")
    parser.add_argument("--source", type=Path, help="YAML with a source block.")
    parser.add_argument("--chunks-dir", type=Path, help="Ranked chunk folder.")
    parser.add_argument("--top", type=int, default=10, help="Draft from the top N chunks.")
    parser.add_argument(
        "--existing",
        type=Path,
        action="append",
        default=[],
        help="Other card files whose cards should not be drafted again.",
    )
    parser.add_argument(
        "--companion",
        type=Path,
        action="append",
        default=[],
        help="A chunk or text file from the same source sent with every excerpt, so cards "
        "may pair statements across chunks (repeatable).",
    )
    parser.add_argument("--no-companion", action="store_true", help="Send no companion file.")
    parser.add_argument(
        "--no-history",
        action="store_true",
        help="Do not skip decisions already proposed by passing drafts in earlier runs.",
    )
    parser.add_argument(
        "--contains",
        action="append",
        default=[],
        help="Only draft chunks containing this text (repeatable; any match).",
    )
    parser.add_argument("--id-prefix", help="Card ID prefix (default: first word of title).")
    parser.add_argument("--run-id", help="Defaults to a timestamp.")
    parser.add_argument("--dry-run", action="store_true", help="Print one payload; no calls.")
    args = apply_book(parser.parse_args())

    cfg = g.load_config(args.config)
    dcfg = cfg.get("drafting", {})
    root = g.ROOT_DIR
    run_id = args.run_id or datetime.now().strftime("drafts%Y%m%d_%H%M%S")
    source, existing = load_source(args.source, include_parked=True)
    for path in args.existing:
        existing += load_source(path, include_parked=True)[1]
    n_cards = len(existing)
    if not args.no_history:
        earlier, runs = history_drafts(
            root / "output" / "card-drafts", source.get("title"), skip={run_id}
        )
        existing += earlier
        print(
            f"Skipping decisions in {n_cards} live/parked card(s) and {len(earlier)} passing "
            f"draft(s) from {len(runs)} earlier run(s)" + (f": {', '.join(runs)}" if runs else "")
        )
    for p in args.companion:
        print(f"Companion: {p}")
    book_text = g.load_file_content(root / source["book_text"])
    prefix = args.id_prefix or slug(str(source.get("title", "")).split()[0])
    chunks = chunks_containing(ranked_chunks(args.chunks_dir, args.top), args.contains)
    if not chunks:
        raise ValueError(f"No ranked chunks (*_rank*_pos*.txt) in {args.chunks_dir} match.")
    companions = [g.load_file_content(p) for p in args.companion]

    if args.dry_run:
        print(build_payload(source, g.load_file_content(chunks[0]), companions))
        return

    prompt_path = drafter_prompt_path(source, dcfg)
    print(f"Drafter prompt: {prompt_path} (mode {source.get('mode')})")
    system_prompt = g.load_file_content(root / prompt_path)
    client = g.make_client(cfg)
    model = dcfg.get("teacher_model") or cfg.get("api", {}).get("teacher_model")
    out_dir = root / "output" / "card-drafts" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    usage.start_log(out_dir)

    drafts, rejected = [], []
    for path in chunks:
        print(f"- {path.name}")
        usage.set_context(chunk=path.name)
        chunk_text = g.load_file_content(path)
        try:
            parsed, raw = gc.call_json(
                client,
                model,
                system_prompt,
                build_payload(
                    source,
                    chunk_text,
                    [
                        t
                        for p, t in zip(args.companion, companions, strict=True)
                        if p.resolve() != path.resolve()
                    ],
                ),
                float(dcfg.get("temperature", 0.2)),
                int(dcfg.get("max_tokens", 16000)),
                dcfg.get("reasoning", {"effort": "medium"}),
            )
        except Exception as e:
            print(f"  Failed: {e.__class__.__name__}: {e}")
            continue
        (out_dir / f"raw_{path.stem}.json").write_text(raw, encoding="utf-8")
        cards, rej = drafts_from_reply(parsed, prefix, path.name)
        for card in cards:
            repairs = repair_card_quotes(card, "\n".join([chunk_text, *companions]), book_text)
            if repairs:
                card["draft"]["quotes_repaired"] = repairs
        print(f"  {len(cards)} draft(s), {len(rej)} rejected")
        drafts += cards
        rejected += rej

    kept, duplicates = dedupe(drafts, existing)
    check_drafts(kept, book_text)
    data = drafts_file(source, kept, rejected, duplicates)
    out_path = out_dir / f"{slug(source.get('title', 'book'))}_drafts.yaml"
    write_drafts(out_path, data)
    n_fail = sum(bool(c["draft"]["issues"]) for c in kept)
    print(
        f"\n{len(kept)} draft(s) ({len(kept) - n_fail} pass the checker, {n_fail} fail), "
        f"{len(duplicates)} duplicate(s) dropped, {len(rejected)} candidate(s) rejected"
    )
    print(f"Drafts: {out_path}")
    usage.finish_log(out_dir, run_id, "draft_cards", root / "output")


if __name__ == "__main__":
    main()
