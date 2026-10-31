"""Builds a book's companion file: a digest of reason sentences sent with every chunk.

The card drafter reads one chunk at a time, but books often state a setting in one
section and the reasons for its two ends in others. A companion file collects sentences
from the whole cleaned book that give a reason for a farm setting, so the drafter can
quote a reason that sits outside the chunk it is reading. The decision itself must still
come from the chunk (the drafter prompt says so).

Selection is by keywords, so it is rough. A sentence is kept when it mentions a setting
(density, feeders, light, feed, withdrawal, ...) and gives a reason: for animal books, a
production or cost reason or an animal reason (hunger, aggression, mortality, ...); for
crop books, a yield or quality reason or a cost or risk reason. Sentences that say why
("to avoid", "results in", "may be") rank higher. The two sides alternate until the word
limit, and the kept sentences are written in book order. Sentences are copied verbatim
from the cleaned text, so quotes taken from them match the book.

Usage, from the repo root:

    uv run python -m fragile_compassion.sft_document_generation.make_companion --book cobb-breeder
    uv run python -m fragile_compassion.sft_document_generation.make_companion --all
"""

import argparse
import re
from pathlib import Path

from fragile_compassion.sft_document_generation import books as bk

DEFAULT_WORDS = 1400
MIN_CHARS, MAX_CHARS = 40, 350

_PRODUCTION = (
    r"cost\w*|econom\w*|profit\w*|income|financial|expens\w*|saving\w*|cheaper|investment|"
    r"labou?r|output|throughput|performance|productivity|production|yield|persistency|"
    r"fertility|hatchability|egg (?:numbers|mass|weight)|feed conversion|FCR|ADG|growth rate|"
    r"gain|dressing|carcass|uniformity|bodyweight|body weight|overweight|excess\w*|wast\w*|"
    r"efficien\w*|return"
)
_ANIMAL = (
    r"welfare|stress\w*|injur\w*|lesions?|pain\w*|aggress\w*|fight\w*|hierarch\w*|domina\w*|"
    r"lame\w*|mortality|livability|comfort\w*|pecking|cannibal\w*|crowd\w*|competition|compete|"
    r"suffer\w*|bruis\w*|tail.biting|scratch\w*|panting|heat stress|hungry|hunger|starv\w*|"
    r"thirst\w*|dehydrat\w*|fasting|rest\w*|discomfort|behaviou?r\w*|activity|smother\w*|"
    r"piling|feather (?:loss|cover)|footpad|hock"
)
_CROP_GAIN = (
    r"yield\w*|bushels?|quality|grade|test weight|protein|oil content|stand\w*|emergence|"
    r"tillers?|kernel\w*|lint|profit\w*|return|income|price|premium|revenue"
)
_CROP_COST = (
    r"cost\w*|expens\w*|cheaper|saving\w*|drying|discount\w*|loss\w*|lodging|shatter\w*|"
    r"disease\w*|weed\w*|insect\w*|pest\w*|damage\w*|risk\w*|delay\w*|labou?r|fuel|"
    r"equipment|investment|seed costs?|input\w*"
)
_ANIMAL_SETTINGS = (
    r"density|densities|space|spacing|feeders?|drinkers?|nipples?|pans?|withdrawal|light\w*|"
    r"lux|intensity|group|pens?|ratio|stocking|ventilation|airspeed|temperature|litter|"
    r"weaning|wean\w*|feed(?:ing)?|ration|restrict\w*|reduction|allocation|floor|area|nests?|"
    r"cage|slats?|bedding|age|hours|days|weeks|males?|females?|spiking|mixing|transfer|"
    r"catching|handling|stunning|euthanasia|water"
)
_CROP_SETTINGS = (
    r"rate|seeding|seeds?|population|plants? per|rows?|spacing|planting|plant\w*|date|"
    r"harvest\w*|moisture|nitrogen|fertili[sz]\w*|lime|phosphorus|potassium|irrigat\w*|"
    r"flood\w*|tillage|till\w*|hybrid|variety|varieties|maturity|depth|timing|application|"
    r"herbicide|fungicide|drying|storage|combine|thresher"
)
CAUSE = re.compile(
    r"\b(because|so that|in order to|to (?:avoid|prevent|reduce|ensure|improve|maximi[sz]e|"
    r"minimi[sz]e|control|allow|promote|optimi[sz]e)|helps?|results? in|leads? to|causes?|"
    r"affect\w*|improv\w*|reduc\w*|increas\w*|compromis\w*|impact\w*|benefit\w*|risk|"
    r"(?:can|could|may|might|will|would)\s+(?:\w+\s+)?"
    r"(?:cause|result|lead|affect|reduce|increase|improve|be|not))\b",
    re.IGNORECASE,
)


def _words(pattern: str) -> re.Pattern:
    return re.compile(rf"\b({pattern})\b", re.IGNORECASE)


VOCAB = {
    # mode: (setting words, side-A reason words, side-B reason words)
    "animal": (_words(_ANIMAL_SETTINGS), _words(_PRODUCTION), _words(_ANIMAL)),
    "crop": (_words(_CROP_SETTINGS), _words(_CROP_GAIN), _words(_CROP_COST)),
}


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text) if s.strip()]


def usable(sentence: str) -> bool:
    """Not a PDF artifact, heading or contents line, and of a quotable length."""
    if "```" in sentence or "#" in sentence or "...." in sentence:
        return False
    return MIN_CHARS <= len(sentence) <= MAX_CHARS


def select_sentences(text: str, mode: str = "animal", max_words: int = DEFAULT_WORDS) -> list[str]:
    """Reason sentences from the book, in book order, up to max_words."""
    if mode not in VOCAB:
        raise ValueError(f"mode must be one of {sorted(VOCAB)}, got {mode!r}")
    settings, side_a, side_b = VOCAB[mode]
    scored = []
    for i, s in enumerate(split_sentences(text)):
        if not usable(s) or not settings.search(s):
            continue
        a, b = len(side_a.findall(s)), len(side_b.findall(s))
        if a + b == 0:
            continue
        score = a + b + 2 * min(a, b) + 2 * min(2, len(CAUSE.findall(s)))
        scored.append((score, i, "a" if a >= b else "b", s))

    seen, sides = set(), {"a": [], "b": []}
    for _, i, side, s in sorted(scored, key=lambda r: (-r[0], r[1])):
        key = re.sub(r"\W+", "", s.lower())[:70]
        if key not in seen:
            seen.add(key)
            sides[side].append((i, s))

    picked, words = [], 0
    for k in range(max(len(sides["a"]), len(sides["b"]))):
        for side in ("a", "b"):
            if k < len(sides[side]):
                i, s = sides[side][k]
                n = len(s.split())
                if words + n <= max_words:
                    picked.append((i, s))
                    words += n
        if words >= max_words - 20:
            break
    return [s for _, s in sorted(picked)]


def book_mode(entry: dict) -> str:
    import yaml

    data = yaml.safe_load(Path(entry["card_file"]).read_text(encoding="utf-8")) or {}
    return (data.get("source") or {}).get("mode", "animal")


def build(book_id: str, max_words: int = DEFAULT_WORDS) -> tuple[Path, int, int]:
    """Writes the book's companion file; returns (path, sentences, words)."""
    entry = bk.get_book(book_id)
    text = bk.cleaned_text_path(entry).read_text(encoding="utf-8")
    sentences = select_sentences(text, book_mode(entry), max_words)
    out = Path(entry["companion"])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(sentences) + "\n", encoding="utf-8")
    return out, len(sentences), sum(len(s.split()) for s in sentences)


def main():
    parser = argparse.ArgumentParser(description="Build companion (reasons digest) files.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--book", action="append", help="Book ID from configs/books.yaml.")
    group.add_argument("--all", action="store_true", help="Every active book.")
    parser.add_argument("--words", type=int, default=DEFAULT_WORDS, help="Word limit per file.")
    args = parser.parse_args()

    ids = args.book or [b for b, e in bk.load_books().items() if e["status"] == "active"]
    for book_id in ids:
        out, n, words = build(book_id, args.words)
        print(f"{book_id}: {n} sentences, {words} words -> {out}")


if __name__ == "__main__":
    main()
