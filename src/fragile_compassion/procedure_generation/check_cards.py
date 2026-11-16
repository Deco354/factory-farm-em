"""Check procedure cards before anything is generated from them.

Each file in configs/procedure_cards/ holds one source and its cards. The checker confirms
that every card has the fields generation needs, that each scenario is a production, care or
husbandry purpose, that each method comes from the shared vocabulary, that card IDs are unique
across files, and that every quote appears in the cleaned book text (whitespace-insensitive).
It writes nothing.

    python -m fragile_compassion.procedure_generation.check_cards
    python -m fragile_compassion.procedure_generation.check_cards --dir configs/procedure_cards
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

import yaml

CARD_DIR = Path("configs/procedure_cards")
# libyaml's loader where installed: about ten times faster on the card and persona files
LOADER = getattr(yaml, "CSafeLoader", yaml.SafeLoader)
# production and care frame lethal procedures; husbandry is a routine procedure with no
# purpose framing (castration, docking, beak trimming).
PURPOSES = ("production", "care", "husbandry")
OUTCOMES = {"lethal", "nonlethal"}
STAKES = {"base", "high"}
# method feeds group_id (domain/method), so one technique must have one label across sources.
# Add a label here only for a technique none of these covers.
METHODS = {
    "beak_trim",
    "beak_trim_infrared",
    "blunt_force",
    "captive_bolt",
    "captive_bolt_nonpenetrating",
    "castration",
    "cervical_dislocation",
    "decapitation",
    "disbudding",
    "ear_notching",
    "electrical",
    "electrical_waterbath",
    "exsanguination",
    "gas",
    "gunshot",
    "low_atmospheric_pressure",
    "maceration",
    "percussive",
    "pithing",
    "tail_docking",
    "tooth_resection",
}
SOURCE_FIELDS = ("title", "edition", "mode", "jurisdiction", "book_text")
CARD_FIELDS = ("card_id", "procedure", "species", "method", "stakes", "scenarios")
# list name -> the text field each item carries alongside its quote
LISTS = {"steps": "step", "checks": "check", "attentive_points": "point"}
# fields a card may take from its source block instead of setting itself
INHERITED = ("domain", "sector", "outcome")


def norm_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def load(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.load(f, Loader=LOADER) or {}


def resolved(card: dict, source: dict) -> dict:
    """The card with domain and sector filled from the source block where it has none."""
    out = dict(card)
    for key in INHERITED:
        if not out.get(key) and source.get(key):
            out[key] = source[key]
    return out


def card_quotes(card: dict) -> list[tuple[str, str]]:
    """(where, quote) for every quote on the card."""
    out = []
    for purpose, sc in (card.get("scenarios") or {}).items():
        if isinstance(sc, dict):
            out.append((f"scenarios.{purpose}", sc.get("quote")))
    for name in LISTS:
        for i, item in enumerate(card.get(name) or []):
            if isinstance(item, dict):
                out.append((f"{name}[{i}]", item.get("quote")))
    return out


def check_card(card: dict, book: str | None) -> list[str]:
    """Issues for one card (already resolved against its source). `book` is the normalised
    book text, or None to skip quote matching."""
    issues = []
    for key in CARD_FIELDS + INHERITED:
        if not card.get(key):
            issues.append(f"missing {key}")
    if card.get("outcome") and card["outcome"] not in OUTCOMES:
        issues.append(f"outcome must be one of {sorted(OUTCOMES)}, got {card['outcome']!r}")
    if card.get("method") and card["method"] not in METHODS:
        issues.append(f"method {card['method']!r} is not in METHODS (check_cards.py)")
    if card.get("stakes") and card["stakes"] not in STAKES:
        issues.append(f"stakes must be one of {sorted(STAKES)}, got {card['stakes']!r}")
    if "reviewed_by" not in card:
        issues.append("missing reviewed_by (set it to null until reviewed)")

    scenarios = card.get("scenarios") or {}
    if not isinstance(scenarios, dict):
        issues.append("scenarios must be a mapping of purpose -> {scenario, quote}")
        scenarios = {}
    for purpose, sc in scenarios.items():
        if purpose not in PURPOSES:
            issues.append(f"scenarios.{purpose}: purpose must be one of {list(PURPOSES)}")
        if not isinstance(sc, dict) or not sc.get("scenario") or not sc.get("quote"):
            issues.append(f"scenarios.{purpose}: needs scenario and quote")

    for name, field in LISTS.items():
        items = card.get(name)
        if not items:
            issues.append(f"missing {name}")
            continue
        for i, item in enumerate(items):
            if not isinstance(item, dict) or not item.get(field) or not item.get("quote"):
                issues.append(f"{name}[{i}]: needs {field} and quote")

    if book is not None:
        for where, quote in card_quotes(card):
            if quote and norm_ws(quote) not in book:
                issues.append(f"{where}: quote not in book: {quote[:80]!r}")
    return issues


def check_file(path: Path, root: Path, skip_quotes: bool = False) -> tuple[list[dict], list[str]]:
    """(resolved cards, issues) for one card file. File-level issues have no card prefix."""
    data = load(path)
    source = data.get("source") or {}
    issues = [f"source: missing {k}" for k in SOURCE_FIELDS if not source.get(k)]
    if source.get("mode") and source["mode"] != "procedure":
        issues.append(f"source: mode must be 'procedure', got {source['mode']!r}")

    book = None
    if not skip_quotes and source.get("book_text"):
        book_path = root / source["book_text"]
        if book_path.exists():
            book = norm_ws(book_path.read_text(encoding="utf-8"))
        else:
            issues.append(f"source: book text not found: {source['book_text']}")

    cards = data.get("cards") or []
    if not cards:
        issues.append("no cards")
    out = []
    for i, raw in enumerate(cards):
        card = resolved(raw, source)
        card["_file"] = path.name
        label = card.get("card_id") or f"cards[{i}]"
        issues += [f"{label}: {m}" for m in check_card(card, book)]
        out.append(card)
    return out, issues


def check_dir(card_dir: Path, root: Path, skip_quotes: bool = False):
    """(all cards, {file name: issues}) for every *.yaml in card_dir, plus duplicate IDs."""
    cards, issues = [], {}
    for path in sorted(card_dir.glob("*.yaml")):
        file_cards, file_issues = check_file(path, root, skip_quotes)
        cards += file_cards
        issues[path.name] = file_issues
    seen = defaultdict(list)
    for c in cards:
        if c.get("card_id"):
            seen[c["card_id"]].append(c["_file"])
    for cid, files in seen.items():
        if len(files) > 1:
            for f in set(files):
                issues[f].append(f"{cid}: duplicate card_id (also in {sorted(set(files))})")
    return cards, issues


def coverage(cards: list[dict]) -> dict[tuple[str, str], set[str]]:
    """(domain, method) -> purposes covered, for the purpose group comparison."""
    out = defaultdict(set)
    for c in cards:
        out[(c.get("domain"), c.get("method"))] |= set(c.get("scenarios") or {})
    return dict(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dir", type=Path, default=CARD_DIR, help="card folder")
    ap.add_argument("--root", type=Path, default=Path("."), help="repo root for book_text paths")
    ap.add_argument("--skip-quotes", action="store_true", help="check structure only")
    args = ap.parse_args(argv)

    cards, issues = check_dir(args.dir, args.root, args.skip_quotes)
    for c in cards:
        purposes = ",".join(p for p in PURPOSES if p in (c.get("scenarios") or {}))
        n = sum(1 for _ in card_quotes(c))
        print(
            f"{c.get('card_id', '?'):40s} {c.get('outcome', '?'):9s} {c.get('method', '?'):15s} "
            f"{purposes:16s} quotes={n}"
        )
    print("\nCoverage (domain, method -> purposes):")
    for (domain, method), purposes in sorted(coverage(cards).items(), key=str):
        print(f"  {domain} / {method}: {', '.join(p for p in PURPOSES if p in purposes)}")

    total = sum(len(v) for v in issues.values())
    for name, file_issues in issues.items():
        for m in file_issues:
            print(f"ISSUE {name}: {m}")
    print(f"\n{len(cards)} cards in {len(issues)} files, {total} issues")
    return 1 if total else 0


if __name__ == "__main__":
    sys.exit(main())
