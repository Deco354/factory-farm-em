"""Moves reviewed drafts from a draft run into the real card file.

Review a run by editing its drafts file (output/card-drafts/<run>/*_drafts.yaml). For each
draft, fix anything that needs fixing in place, then set:

    decision: keep    # move into `cards:` with reviewed_by set
    decision: park    # move into `parked:`; also set `why:` (required)
    decision: skip    # leave it out (the default for drafts with no decision)

Then promote:

    uv run python -m fragile_compassion.sft_document_generation.promote_drafts \\
        --run cobb-breeder-comp-02 --reviewer CB

The card file comes from configs/books.yaml (matched by the drafts' source title), or
from --book when several card files share a book (the AVMA guidelines). Kept drafts are
checked again first, so a draft fixed by hand is judged on its current text; one that
fails is not moved and the run reports why. Drafts already in the card file (same
card_id, or the same decision) are not added twice, so promoting a run again is safe.

The card file's existing text, comments and layout are left as they are: new cards are
appended to the end of `cards:` and `parked:`. Use --dry-run to see what would happen.
"""

import argparse
import re
from pathlib import Path
from typing import Any

from fragile_compassion.sft_document_generation import books as bk
from fragile_compassion.sft_document_generation import cards as cd
from fragile_compassion.sft_document_generation import draft_cards as dc
from fragile_compassion.sft_document_generation import generate as g

DECISIONS = ("keep", "park", "skip")
REVIEW_KEYS = ("draft", "decision", "why")
LEGAL_KEYS = ("legal_end", "legal_quote", "legal_wording")  # empty on non-legal drafts
PARKED_HEADER = "# Parked: kept for reference, not generated (the loader reads only `cards`)."


# ---------------------------------------------------------------------------
# Reading a run
# ---------------------------------------------------------------------------


def drafts_path(run_dir: Path) -> Path:
    found = sorted(Path(run_dir).glob("*_drafts.yaml"))
    if len(found) != 1:
        raise ValueError(f"{run_dir}: expected one *_drafts.yaml, found {len(found)}.")
    return found[0]


def load_drafts(path: Path) -> tuple[dict, list[dict]]:
    import yaml

    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(data.get("source"), dict):
        raise TypeError(f"{path}: expected a top-level 'source' mapping.")
    return data["source"], [c for c in data.get("cards") or [] if isinstance(c, dict)]


def decision_of(draft: dict) -> str:
    value = draft.get("decision")
    decision = str(value).strip().lower() if value else "skip"
    if decision not in DECISIONS:
        raise ValueError(
            f"{draft.get('card_id')}: decision must be one of {DECISIONS}, got {value!r}."
        )
    return decision


def card_file_for(title: str, books_path: Path = bk.BOOKS_PATH, root: Path = g.ROOT_DIR) -> Path:
    """The one registry card file whose source title matches the drafts'."""
    matches = []
    for book_id in bk.load_books(books_path):
        entry = bk.get_book(book_id, books_path, root)
        if Path(entry["card_file"]).exists():
            source, _ = dc.load_source(entry["card_file"])
            if source.get("title") == title:
                matches.append((book_id, entry["card_file"]))
    if len(matches) != 1:
        found = ", ".join(b for b, _ in matches) or "none"
        raise ValueError(f"Card files for {title!r} in the registry: {found}. Use --book.")
    return matches[0][1]


# ---------------------------------------------------------------------------
# Deciding what moves
# ---------------------------------------------------------------------------


def clean_card(draft: dict, decision: str, reviewer: str | None) -> dict:
    """The draft as it should appear in the card file."""
    card = {
        k: v
        for k, v in draft.items()
        if k not in REVIEW_KEYS and not (k in LEGAL_KEYS and v is None)
    }
    if decision == "keep":
        card["reviewed_by"] = card.get("reviewed_by") or reviewer
    else:  # park: the reason goes right after the ID, as in hand-parked cards
        why = " ".join(str(draft.get("why") or "").split())
        card = {"card_id": card.pop("card_id"), "why": why, **card}
    return card


def plan(
    drafts: list[dict], card_data: dict, book_text: str, reviewer: str | None
) -> dict[str, list]:
    """{keep, park: cards to add; skip, already, failed: (card_id, note)}."""
    live, parked = card_data.get("cards") or [], card_data.get("parked") or []
    ids = {c.get("card_id") for c in live + parked}
    keys = {dc.draft_key(c) for c in live + parked}
    result: dict[str, list] = {"keep": [], "park": [], "skip": [], "already": [], "failed": []}
    for draft in drafts:
        card_id, decision = draft.get("card_id"), decision_of(draft)
        if decision == "skip":
            result["skip"].append((card_id, ""))
            continue
        if card_id in ids or dc.draft_key(draft) in keys:
            result["already"].append((card_id, "already in the card file"))
            continue
        if decision == "park" and not str(draft.get("why") or "").strip():
            result["failed"].append((card_id, "park needs a `why:`"))
            continue
        if decision == "keep":
            if not (draft.get("reviewed_by") or reviewer):
                result["failed"].append((card_id, "keep needs --reviewer or reviewed_by"))
                continue
            issues = cd.check_card(draft, book_text)
            if issues:
                result["failed"].append((card_id, "; ".join(issues)))
                continue
        result[decision].append(clean_card(draft, decision, reviewer))
        ids.add(card_id)
        keys.add(dc.draft_key(draft))
    return result


# ---------------------------------------------------------------------------
# Writing into the card file without disturbing it
# ---------------------------------------------------------------------------


def indented_dumper():
    """A yaml SafeDumper that indents lists inside mappings, like the hand-written files."""
    import yaml

    class Dumper(yaml.SafeDumper):
        def increase_indent(self, flow=False, indentless=False):
            return super().increase_indent(flow, False)

    return Dumper


def card_block(card: dict, item_indent: int) -> str:
    """One card as a YAML list item, indented to sit under its section."""
    import yaml

    text = yaml.dump(
        [card],
        Dumper=indented_dumper(),
        sort_keys=False,
        allow_unicode=True,
        width=100,
        default_flow_style=False,
    )
    pad = " " * item_indent
    return "".join(pad + line if line.strip() else line for line in text.splitlines(True))


def section_bounds(lines: list[str], key: str) -> tuple[int, int] | None:
    """(index of the `key:` line, index where its items end) or None if absent.

    A section ends at the next top-level key. Blank lines and top-level comments just
    before that key belong to the next section, so new items go above them."""
    start = next((i for i, ln in enumerate(lines) if re.match(rf"{key}:\s*(\[\])?\s*$", ln)), None)
    if start is None:
        return None
    end = len(lines)
    for i in range(start + 1, len(lines)):
        ln = lines[i]
        if ln.strip() and not ln[0].isspace() and not ln.startswith(("-", "#")):
            end = i
            break
    while end > start + 1 and (not lines[end - 1].strip() or lines[end - 1].startswith("#")):
        end -= 1
    return start, end


def item_indent(lines: list[str], start: int, end: int) -> int:
    """The indent of the section's existing list items (2 if it has none)."""
    for ln in lines[start + 1 : end]:
        m = re.match(r"( *)- ", ln)
        if m:
            return len(m.group(1))
    return 2


def insert_items(text: str, key: str, cards: list[dict]) -> str:
    """The card file text with `cards` appended to the `key` section (made if missing)."""
    if not cards:
        return text
    lines = text.splitlines(True)
    if lines and not lines[-1].endswith("\n"):
        lines[-1] += "\n"
    bounds = section_bounds(lines, key)
    if bounds is None:
        header = [f"\n{PARKED_HEADER}\n"] if key == "parked" else ["\n"]
        lines += [*header, f"{key}:\n"]
        bounds = (len(lines) - 1, len(lines))
    start, end = bounds
    lines[start] = f"{key}:\n"  # `cards: []` becomes a list with items
    indent = item_indent(lines, start, end)
    block = "".join(card_block(c, indent) for c in cards)
    lines[end:end] = [block]
    return "".join(lines)


def write_card_file(path: Path, keep: list[dict], park: list[dict]) -> None:
    """Appends the cards, keeping the file's line endings, and checks the result loads."""
    import yaml

    raw = Path(path).read_bytes().decode("utf-8")
    newline = "\r\n" if "\r\n" in raw else "\n"
    text = insert_items(raw.replace("\r\n", "\n"), "cards", keep)
    text = insert_items(text, "parked", park)
    data = yaml.safe_load(text) or {}
    before = yaml.safe_load(raw) or {}
    for key, added in (("cards", keep), ("parked", park)):
        if len(data.get(key) or []) != len(before.get(key) or []) + len(added):
            raise RuntimeError(f"{path}: `{key}` did not come out as expected; nothing written.")
    Path(path).write_bytes(text.replace("\n", newline).encode("utf-8"))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def report(result: dict[str, list], card_file: Path, dry_run: bool) -> None:
    verb = "Would add" if dry_run else "Added"
    print(f"Card file: {card_file}")
    for key, label in (("keep", "to cards"), ("park", "to parked")):
        for c in result[key]:
            print(f"  {verb} {label}: {c['card_id']}")
    for key, label in (("already", "Already there"), ("failed", "NOT MOVED"), ("skip", "Skipped")):
        for card_id, note in result[key]:
            print(f"  {label}: {card_id}" + (f" ({note})" if note else ""))
    counts = ", ".join(
        f"{len(result[k])} {k}" for k in ("keep", "park", "skip", "already", "failed")
    )
    print(f"\n{counts}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Move reviewed drafts into a card file.")
    parser.add_argument("--run", required=True, help="Run folder name under output/card-drafts/.")
    parser.add_argument("--book", help="Book ID from configs/books.yaml (if the title is shared).")
    parser.add_argument("--reviewer", help="Initials for reviewed_by on kept cards.")
    parser.add_argument("--dry-run", action="store_true", help="Report only; change nothing.")
    args = parser.parse_args()

    root = g.ROOT_DIR
    run_dir = root / "output" / "card-drafts" / args.run
    source, drafts = load_drafts(drafts_path(run_dir))
    card_file = (
        bk.get_book(args.book)["card_file"] if args.book else card_file_for(source.get("title"))
    )
    card_data: dict[str, Any] = cd.load_card_file(card_file)
    book_text = (root / card_data["source"]["book_text"]).read_text(encoding="utf-8")

    result = plan(drafts, card_data, book_text, args.reviewer)
    if not args.dry_run and (result["keep"] or result["park"]):
        write_card_file(card_file, result["keep"], result["park"])
        bad = {k: v for k, v in cd.check_card_file(card_file, root).items() if v}
        if bad:
            print(f"Note: the card file now has checker issues: {sorted(bad)}")
    report(result, card_file, args.dry_run)
    return 1 if result["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
