"""Parameter cards: hand-curated trade-off parameters for the cards pipeline.

A card file (YAML) describes one source book and the parameters a researcher has
chosen from it. Each card fixes the range, which end is the output end, and the
source of every reason. The generator never decides these things; it only writes
prompts and responses for cards that pass `check_card` and have been reviewed.

Reasons: every reason must have a basis, recorded per end.
  quote  The reason is a close paraphrase of that end's quote (output_end_quote or
         other_end_quote). Numbers in it must come from the card's quotes. Whether
         the paraphrase adds anything is for the reviewer to judge.
  unit   Only for space_allowance and animals_per_equipment. The reason field is left
         empty and a fixed sentence restating the unit is used instead (UNIT_TEMPLATES),
         so no free wording enters the data.

Card file layout:

    source:
      title: "Cobb Broiler Management Guide"
      edition: "2021"
      domain: poultry_production        # poultry_production | swine_production | crop_agronomy
      mode: animal                      # animal | crop
      jurisdiction: US
      book_text: "context/cleaned/<stem>_cleaned.txt"   # cleaned text the quotes come from

    cards:
      - card_id: cobb-density-tunnel-evap
        parameter: "..."
        parameter_category: space_allowance   # space_allowance | animals_per_equipment |
                                              # environment | feeding | feed_level |
                                              # procedure_timing | other
        unit: "kg/m²"
        range_low: 28
        range_high: 42
        range_quote: "..."              # string, or list if the ends are in different sentences
        output_end: high                # low | high
        output_reason_basis: unit       # quote | unit
        output_end_reason: null         # required for quote basis; must be empty for unit basis
        output_end_quote: null          # required for quote basis
        other_reason_basis: quote       # quote | unit
        other_end_reason: "..."
        other_end_quote: "..."
        management_conditions:          # optional; applies_to: low | high | both
          - {condition: "...", applies_to: both, quote: "..."}
        production_consequences:        # optional
          - {consequence: "...", applies_to: high, quote: "..."}
        page: 40                        # optional
        notes: "..."                    # optional
        reviewed_by: null               # initials of the second reader; unreviewed cards
                                        # are not generated unless --allow-unreviewed

Check a card file from the repo root:

    uv run python -m fragile_compassion.sft_document_generation.cards configs/cards/<file>.yaml
"""

import argparse
import sys
from pathlib import Path
from typing import Any

from fragile_compassion.sft_document_generation import generate as g

REQUIRED_FIELDS = [
    "card_id",
    "parameter",
    "parameter_category",
    "unit",
    "range_low",
    "range_high",
    "range_quote",
    "output_end",
    "output_reason_basis",
    "other_reason_basis",
]
REQUIRED_SOURCE_FIELDS = ["title", "edition", "domain", "mode", "jurisdiction", "book_text"]
REASON_BASES = {"quote", "unit"}

# Fixed reason sentences for unit-basis reasons. Keys: (category, "output" | "other").
UNIT_TEMPLATES = {
    ("space_allowance", "output"): "At {value} {unit}, the same floor area carries more stock.",
    ("space_allowance", "other"): "At {value} {unit}, the stock has more floor area.",
    ("animals_per_equipment", "output"): "At {value} {unit}, each piece of equipment serves "
    "more animals.",
    ("animals_per_equipment", "other"): "At {value} {unit}, each piece of equipment serves "
    "fewer animals.",
    ("feed_level", "output"): "At {value} {unit}, the animals receive less of their full feed.",
    ("feed_level", "other"): "At {value} {unit}, the animals receive more of their full feed.",
}
# Fixed consequence sentences, used only to fill the end that has no quoted consequence
# when the other end has one, so both arms state a cost of their choice.
UNIT_CONSEQUENCE_TEMPLATES = {
    ("space_allowance", "output"): "At {value} {unit}, the stock has less floor area.",
    ("space_allowance", "other"): "At {value} {unit}, the same floor area carries less stock.",
    ("animals_per_equipment", "output"): "At {value} {unit}, more animals share each "
    "piece of equipment.",
    ("animals_per_equipment", "other"): "At {value} {unit}, more pieces of equipment are "
    "needed for the same number of animals.",
}
SIDE_FIELDS = {
    "output": ("output_reason_basis", "output_end_reason", "output_end_quote"),
    "other": ("other_reason_basis", "other_end_reason", "other_end_quote"),
}


# ---------------------------------------------------------------------------
# Loading and shaping
# ---------------------------------------------------------------------------


def load_card_file(path: Path) -> dict[str, Any]:
    import yaml

    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(data.get("cards"), list) or not isinstance(data.get("source"), dict):
        raise ValueError(f"{path}: expected top-level 'source' (mapping) and 'cards' (list).")
    return data


def as_list(value) -> list[str]:
    if value is None:
        return []
    return [value] if isinstance(value, str) else [v for v in value if v]


def card_quotes(card: dict) -> list[str]:
    """Every quote on the card, range quotes first and in order."""
    quotes = as_list(card.get("range_quote"))
    quotes += as_list(card.get("output_end_quote")) + as_list(card.get("other_end_quote"))
    for key in ("management_conditions", "production_consequences"):
        quotes += [
            c.get("quote") for c in card.get(key) or [] if isinstance(c, dict) and c.get("quote")
        ]
    return quotes


def quotes_text(card: dict) -> str:
    """All of a card's quotes as one text. Used as the 'excerpt' for response checks."""
    return "\n".join(card_quotes(card))


def end_value(card: dict, end: str) -> float:
    return float(card["range_low"] if end == "low" else card["range_high"])


def side_end(card: dict, side: str) -> str:
    """'low' or 'high' for side 'output' or 'other'."""
    out = card["output_end"]
    return out if side == "output" else {"low": "high", "high": "low"}[out]


def resolved_reason(card: dict, side: str) -> str | None:
    """The reason text the response writer receives for this side of the range."""
    basis_key, reason_key, _ = SIDE_FIELDS[side]
    if card.get(basis_key) == "unit":
        template = UNIT_TEMPLATES.get((card.get("parameter_category"), side))
        if template is None:
            return None
        value = g.canon_num(end_value(card, side_end(card, side)))
        return template.format(value=value, unit=card.get("unit", ""))
    return card.get(reason_key)


def card_to_param(card: dict) -> dict:
    """Shapes a card like a v2 parameter so generate.check_parameter/check_item apply."""
    param = {k: v for k, v in card.items() if k not in {"page", "curator", "notes", "reviewed_by"}}
    param["param_key"] = card["card_id"]
    param["range_quote"] = " ".join(as_list(card.get("range_quote")))
    # Reason quotes may be lists; generate.check_parameter expects strings.
    for key in ("output_end_quote", "other_end_quote"):
        param[key] = " ".join(as_list(card.get(key))) or None
    param["output_basis"] = "definitional" if card.get("output_reason_basis") == "unit" else "quote"
    param["output_end_reason"] = resolved_reason(card, "output")
    param["other_end_reason"] = resolved_reason(card, "other")
    param["management_conditions"] = card.get("management_conditions") or []
    param["production_consequences"] = card.get("production_consequences") or []
    return param


def end_of(card: dict, arm: str) -> str:
    """'low' or 'high': which end of the range this arm recommends."""
    return side_end(card, "other" if arm in g.OTHER_END_ARMS else "output")


def items_for_end(entries: list[dict] | None, end: str) -> list[dict]:
    """Conditions or consequences that apply to the given end."""
    return [e for e in entries or [] if e.get("applies_to") in (end, "both")]


def consequences_for(card: dict, end: str) -> list[dict]:
    """Quoted consequences for this end, plus a fixed unit sentence when this end has
    none but the other end does (definitional categories only)."""
    quoted = items_for_end(card.get("production_consequences"), end)
    if quoted:
        return quoted
    other_end = "low" if end == "high" else "high"
    if not items_for_end(card.get("production_consequences"), other_end):
        return []
    side = "output" if end == card["output_end"] else "other"
    template = UNIT_CONSEQUENCE_TEMPLATES.get((card.get("parameter_category"), side))
    if template is None:
        return []
    text = template.format(value=g.canon_num(end_value(card, end)), unit=card.get("unit", ""))
    return [{"consequence": text, "applies_to": end, "quote": None, "basis": "unit"}]


def is_reviewed(card: dict) -> bool:
    return bool(str(card.get("reviewed_by") or "").strip())


# ---------------------------------------------------------------------------
# Checking
# ---------------------------------------------------------------------------


def check_reason(card: dict, side: str) -> list[str]:
    basis_key, reason_key, quote_key = SIDE_FIELDS[side]
    basis, reason, quote = card.get(basis_key), card.get(reason_key), card.get(quote_key)
    category = card.get("parameter_category")
    issues = []
    if basis not in REASON_BASES:
        return [f"{basis_key} must be 'quote' or 'unit', got {basis!r}"]
    if basis == "quote":
        if not quote:
            issues.append(f"{basis_key} is 'quote' but {quote_key} is empty")
        if not reason:
            issues.append(f"{basis_key} is 'quote' but {reason_key} is empty")
        else:
            allowed = set()
            for q in card_quotes(card):
                allowed |= g.numbers_in(q, words=True)
            stray = g.numbers_in(reason) - allowed
            if stray:
                issues.append(f"{reason_key} has numbers not in the card's quotes: {sorted(stray)}")
    else:
        if (category, side) not in UNIT_TEMPLATES:
            issues.append(f"{basis_key} 'unit' not allowed for category {category!r}")
        if reason:
            issues.append(
                f"{reason_key} must be empty when {basis_key} is 'unit' "
                f"(the fixed sentence is used)"
            )
    return issues


def check_card(card: dict, book_text: str) -> list[str]:
    issues = [f"missing field: {f}" for f in REQUIRED_FIELDS if card.get(f) in (None, "", [])]
    if issues:
        return issues

    book_n = g.norm_ws(book_text)
    for q in card_quotes(card):
        if g.norm_ws(q) not in book_n:
            issues.append(f"quote not found in book text: {q[:80]!r}")

    issues += check_reason(card, "output") + check_reason(card, "other")
    issues += [
        i
        for i in g.check_parameter(card_to_param(card), quotes_text(card))
        if "output_end_quote is empty" not in i and "output_basis" not in i
    ]

    for key, label in (
        ("management_conditions", "condition"),
        ("production_consequences", "consequence"),
    ):
        for entry in card.get(key) or []:
            if entry.get("applies_to") not in {"low", "high", "both"}:
                issues.append(f"{label} applies_to must be low, high or both: {entry.get(label)!r}")

    low, high = consequences_for(card, "low"), consequences_for(card, "high")
    if bool(low) != bool(high):
        missing = "low" if not low else "high"
        issues.append(
            f"consequences stated for only one end; add a quoted consequence for the {missing} end "
            "so both arms state a cost of their choice"
        )
    return issues


def end_specific_count(card: dict, end: str) -> int:
    """Conditions and consequences that apply only to this end (not 'both')."""
    conds = [c for c in card.get("management_conditions") or [] if c.get("applies_to") == end]
    cons = [c for c in consequences_for(card, end) if c.get("applies_to") == end]
    return len(conds) + len(cons)


def balance_warnings(card: dict) -> list[str]:
    """Warns when one arm gets noticeably more end-specific content than the other,
    which makes its responses longer. Does not fail the card."""
    low, high = end_specific_count(card, "low"), end_specific_count(card, "high")
    if abs(low - high) > 1:
        return [
            f"unbalanced content: {high} end-specific items for the high end vs"
            f" {low} for the low end; "
            "merge or trim items so the arms differ by at most one"
        ]
    return []


def check_card_file(path: Path, root: Path) -> dict[str, list[str]]:
    """Returns {card_id or '<file>': [issues]} for every card, plus file-level issues."""
    data = load_card_file(path)
    src = data["source"]
    report: dict[str, list[str]] = {}

    file_issues = [f"source missing field: {f}" for f in REQUIRED_SOURCE_FIELDS if not src.get(f)]
    if src.get("mode") not in g.ARMS_BY_MODE:
        file_issues.append(f"source.mode must be one of {list(g.ARMS_BY_MODE)}")
    ids = [c.get("card_id") for c in data["cards"]]
    dupes = sorted({i for i in ids if ids.count(i) > 1})
    if dupes:
        file_issues.append(f"duplicate card_id: {dupes}")

    book_path = root / src.get("book_text", "")
    if not book_path.is_file():
        file_issues.append(f"book_text not found: {book_path}")
        report["<file>"] = file_issues
        return report
    book_text = book_path.read_text(encoding="utf-8")
    report["<file>"] = file_issues

    for card in data["cards"]:
        report[card.get("card_id") or "<no id>"] = check_card(card, book_text)
    return report


def main():
    parser = argparse.ArgumentParser(
        description="Check a parameter card file against its book text."
    )
    parser.add_argument("card_file", type=Path)
    parser.add_argument(
        "--root", type=Path, default=g.ROOT_DIR, help="Repo root for book_text paths."
    )
    args = parser.parse_args()

    report = check_card_file(args.card_file, args.root)
    cards = {c.get("card_id"): c for c in load_card_file(args.card_file)["cards"]}
    for i in report.get("<file>", []):
        print(f"FILE  - {i}")
    passed = reviewed = 0
    for key, issues in report.items():
        if key == "<file>":
            continue
        card = cards.get(key, {})
        status = "FAIL" if issues else ("OK" if is_reviewed(card) else "OK, NOT REVIEWED")
        passed += not issues
        reviewed += (not issues) and is_reviewed(card)
        print(f"{status:17} {key}")
        for i in issues:
            print(f"      - {i}")
        if not issues:
            for w in balance_warnings(card):
                print(f"      WARN: {w}")
            for side in ("output", "other"):
                print(
                    f"      {side} reason ({card.get(SIDE_FIELDS[side][0])}): "
                    f"{resolved_reason(card, side)}"
                )
            for side in ("output", "other"):
                for c in consequences_for(card, side_end(card, side)):
                    tag = "unit" if c.get("basis") == "unit" else "quote"
                    print(f"      {side} consequence ({tag}): {c['consequence']}")
    total = len(report) - 1
    print(
        f"\n{passed}/{total} cards pass checks; {reviewed} of those are reviewed and will "
        f"be generated."
    )
    sys.exit(1 if (passed < total or report.get("<file>")) else 0)


if __name__ == "__main__":
    main()
