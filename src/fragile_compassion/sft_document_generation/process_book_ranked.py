"""Clean, chunk, and rank book excerpts for SFT data generation.

Version 2 of the chunker. Changes from v1:
- Adds `calculate_relevance_score`, which rewards stated numeric ranges/bounds in
  husbandry units (days, hours, lux, °C, kg/m², ...) and husbandry vocabulary, and
  penalises drug/vaccine dosing language. v1's density score favoured dosing and
  nutrition units and counted citation years.
- Chunks are ranked by relevance and the top N are exported (instead of keeping
  everything above a low threshold). Filenames carry the rank.
- Writes a manifest (JSON) with every chunk's score breakdown, original position,
  and text hash, so selection can be inspected and records traced.
- Seeds the workspace buffer with the highest-ranked chunk, not the first one.

Cleaning and chunking logic are unchanged from v1. The v1 density functions are
kept for compatibility with any code that imports them.
"""

import hashlib
import json
import re
from pathlib import Path
from typing import Any

# Path routing relative to repository root
PROCESS_DIR = Path(__file__).resolve().parent
SRC_DIR = PROCESS_DIR.parent.parent
ROOT_DIR = SRC_DIR.parent
DEFAULT_CONFIG_PATH = ROOT_DIR / "configs" / "sft_doc_config.yaml"

CHUNKER_VERSION = "2.0"


# ---------------------------------------------------------------------------
# Relevance scoring (new in v2)
# ---------------------------------------------------------------------------

# Order matters inside the alternation: longer units first (kg/m2 before kg).
UNITS = (
    r"(?:%|kg/m2|kg/m²|birds?/m2|birds?/m²|ft2|ft²|sq\.? ?ft|lux|lx|°C|°F|"
    r"hours?|hrs?|h|days?|d|weeks?|wks?|g|kg|lbs?|cm|mm|m|ppm)"
)
NUM = r"\d+(?:\.\d+)?"

# "30-39 kg/m2", "21 to 28 days". The cleaner already normalises en/em dashes to "-".
RANGE_RE = re.compile(rf"\b{NUM}\s*(?:-|to)\s*{NUM}\s*{UNITS}(?!\w)", re.IGNORECASE)

# "a minimum of 4 hours", "should not exceed 39 kg/m2", "at least 0.5 m"
BOUND_RE = re.compile(
    rf"\b(?:minimum|maximum|at least|not exceed|no more than|up to)\b"
    rf"[^.\n]{{0,40}}?{NUM}\s*{UNITS}(?!\w)",
    re.IGNORECASE,
)

HUSBANDRY_RE = re.compile(
    r"\b(?:stocking|density|lighting|darkness|photoperiod|lux|wean|weaning|"
    r"feed restriction|beak|trim|crate|stall|gestation|farrowing|catching|"
    r"transport|feed withdrawal|space allowance|perch|litter|ammonia|lameness|"
    r"gait|footpad)\w*",
    re.IGNORECASE,
)

PHARMA_RE = re.compile(
    r"(?:\bmg/kg\b|\bIU\b|\bmL/L\b|\bdose\b|\bdosage\b|\bantibiotic\w*|"
    r"\bantimicrobial\w*|\bvaccin\w*|\bwithdrawal period\b)",
    re.IGNORECASE,
)

PIPE_ROW_RE = re.compile(r"(?m)^\|.+\|$")


def relevance_breakdown(text: str) -> dict[str, float]:
    """Returns the components of the relevance score for a chunk."""
    ranges = len(RANGE_RE.findall(text)) + len(BOUND_RE.findall(text))
    husbandry = len(HUSBANDRY_RE.findall(text))
    pharma = len(PHARMA_RE.findall(text))
    table_rows = len(PIPE_ROW_RE.findall(text))

    score = (
        3.0 * min(ranges, 10)
        + 1.0 * min(husbandry, 15)
        + 0.5 * min(table_rows, 20)
        - 2.0 * min(pharma, 10)
    )
    return {
        "score": score,
        "ranges": ranges,
        "husbandry_terms": husbandry,
        "pharma_terms": pharma,
        "table_rows": table_rows,
    }


def calculate_relevance_score(text: str) -> float:
    """Scores how likely a chunk is to contain a husbandry parameter with a stated range."""
    return relevance_breakdown(text)["score"]


# ---------------------------------------------------------------------------
# v1 density scoring (kept for compatibility; no longer used for selection)
# ---------------------------------------------------------------------------


def calculate_data_density_score(text: str) -> int:
    """Calculates a numerical data density score for a text chunk (v1)."""
    score = 0.0

    pipe_rows = len(re.findall(r"(?m)^\|.+\|$", text))
    if pipe_rows > 1:
        score += pipe_rows * 2.0

    quant_matches = len(
        re.findall(
            r"\b\d+(?:\.\d+)?\s*(?:%|mg|g|kg|IU|ppm|kcal|mm|cm|m)(?!\w)", text, re.IGNORECASE
        )
    )
    score += min(quant_matches, 10) * 2.0

    bare_nums = len(re.findall(r"\b\d+(?:\.\d+)?\b", text)) - quant_matches
    score += min(max(0, bare_nums), 10) * 0.5

    return int(score)


def is_data_dense_chunk(text: str, min_score: int = 5) -> bool:
    """Evaluates whether a text chunk exceeds the v1 data density threshold."""
    return calculate_data_density_score(text) >= min_score


# ---------------------------------------------------------------------------
# Quality filter, config, chunking, cleaning (unchanged from v1)
# ---------------------------------------------------------------------------


def is_quality_content_chunk(text: str) -> bool:
    """Filters out front-matter, TOC, copyright notices, and low-density text."""
    front_matter_patterns = [
        r"ISBN\s+978",
        r"Library of Congress Cataloging",
        r"Set in \d+/\d+pt",
        r"Hb printing \d+",
        r"Downloaded From",
        r"Contributors\s+xviii",
        r"Editor‐in‐Chief",
    ]
    for pattern in front_matter_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            matches = sum(1 for p in front_matter_patterns if re.search(p, text, re.IGNORECASE))
            if matches >= 2:
                return False

    toc_lines = len(re.findall(r"(?m)^.+?(?:\.{2,}|\s{2,})\s*\d+\s*$", text))
    if toc_lines > 5:
        return False

    words = text.split()
    if len(words) < 200:
        return False

    return True


def load_config(config_path: Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    """Loads configuration settings lazily from YAML."""
    try:
        import yaml
    except ImportError as e:
        raise ImportError("The 'PyYAML' package is required to parse YAML configs.") from e

    if not config_path.exists():
        example_path = config_path.with_name("sft_doc_config.yaml.example")
        raise FileNotFoundError(
            f"Configuration file not found at: {config_path}\n"
            f"Please copy '{example_path}' to '{config_path}'."
        )

    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def chunk_text_by_words(text: str, chunk_size: int = 1500, min_tail_words: int = 300) -> list[str]:
    """Splits cleaned text into chunks of approximately `chunk_size` words.

    Merges trailing remainders under `min_tail_words` into the final chunk to avoid
    emitting undersized tail chunks that fail quality filters.
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []
    current_paragraphs: list[str] = []
    current_word_count = 0

    for paragraph in paragraphs:
        paragraph_word_count = len(paragraph.split())

        if paragraph_word_count > chunk_size:
            if current_paragraphs:
                chunks.append("\n\n".join(current_paragraphs))
                current_paragraphs = []
                current_word_count = 0

            sentences = re.split(r"(?<=[.!?])\s+", paragraph)
            sub_chunk: list[str] = []
            sub_word_count = 0

            for sentence in sentences:
                s_words = len(sentence.split())
                if sub_word_count + s_words > chunk_size and sub_chunk:
                    chunks.append(" ".join(sub_chunk))
                    sub_chunk = [sentence]
                    sub_word_count = s_words
                else:
                    sub_chunk.append(sentence)
                    sub_word_count += s_words

            if sub_chunk:
                current_paragraphs = [" ".join(sub_chunk)]
                current_word_count = sub_word_count

        elif current_word_count + paragraph_word_count > chunk_size and current_paragraphs:
            chunks.append("\n\n".join(current_paragraphs))
            current_paragraphs = [paragraph]
            current_word_count = paragraph_word_count
        else:
            current_paragraphs.append(paragraph)
            current_word_count += paragraph_word_count

    if current_paragraphs:
        tail_text = "\n\n".join(current_paragraphs)
        if chunks and len(tail_text.split()) < min_tail_words:
            chunks[-1] = chunks[-1] + "\n\n" + tail_text
        else:
            chunks.append(tail_text)

    return chunks


def clean_markdown_text(text: str) -> str:
    """Clean, robust layout-based cleaner for PDF-extracted Markdown text."""
    text = re.sub(r"(?m)^\s*(?:\d+\s+[A-Z\s]{3,}|[A-Z\s]{3,}\s+\d+)\s*$", "", text)
    text = re.sub(r"!\[\]\(\)", "", text)
    text = re.sub(r"\\([.!\-])", r"\1", text)
    text = re.sub(
        r"(?m)^[ \t]*(?:\d+(?:\.\d+)?[ \t]+){2,}\d+(?:\.\d+)?(?:[ \t]+[A-Z]{1,4})*[ \t]*$",
        "",
        text,
    )
    text = re.sub(r"(\w+)-\s*\n\s*(\w+)", r"\1\2", text)
    text = re.sub(r"(?<![:.\-\n])\n(?!\n|[A-Z0-9\-\*#|])", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"(\d+)\s*[–—\-]\s*(\d+)", r"\1-\2", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


# ---------------------------------------------------------------------------
# Main pipeline (selection logic changed in v2)
# ---------------------------------------------------------------------------


def process_book_file(
    input_file: Path | None = None,
    output_file: Path | None = None,
    chunk_size_words: int | None = None,
    config_path: Path = DEFAULT_CONFIG_PATH,
) -> Path:
    """Reads raw source markdown, cleans it, chunks it, ranks chunks by relevance,
    exports the top N, writes a manifest, and seeds context/excerpt.txt with the
    highest-ranked chunk.

    Config keys (under `defaults`):
      raw_book_path, chunks_dir, context_path, chunk_size_words   (as in v1)
      top_n_chunks          int, default 40. Number of ranked chunks to export.
      min_relevance_score   float, default 1.0. Chunks below this are never exported.
    """
    cfg = load_config(config_path).get("defaults", {})

    if input_file is None:
        input_file = ROOT_DIR / cfg.get("raw_book_path", "context/raw/sample_book.md")

    if chunk_size_words is None:
        chunk_size_words = cfg.get("chunk_size_words", 1500)

    top_n = int(cfg.get("top_n_chunks", 40))
    min_relevance = float(cfg.get("min_relevance_score", 1.0))

    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    raw_text = input_file.read_text(encoding="utf-8")
    cleaned_text = clean_markdown_text(raw_text)

    # 1. Export full cleaned text
    if output_file is None:
        cleaned_dir = ROOT_DIR / "context" / "cleaned"
        full_cleaned_path = cleaned_dir / f"{input_file.stem}_cleaned.txt"
    else:
        full_cleaned_path = output_file

    full_cleaned_path.parent.mkdir(parents=True, exist_ok=True)
    full_cleaned_path.write_text(cleaned_text, encoding="utf-8")
    print(f"Successfully processed full text: {input_file.name} -> {full_cleaned_path}")

    # 2. Chunk, filter noise, score
    chunks_dir = ROOT_DIR / cfg.get("chunks_dir", "context/chunks") / input_file.stem
    chunks_dir.mkdir(parents=True, exist_ok=True)

    print(f"Chunking into ~{chunk_size_words}-word excerpts...")
    raw_chunks = chunk_text_by_words(cleaned_text, chunk_size=chunk_size_words)

    candidates: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []

    for position, chunk in enumerate(raw_chunks, start=1):
        row: dict[str, Any] = {
            "original_position": position,  # position in the book before any filtering
            "sha256": hashlib.sha256(chunk.encode("utf-8")).hexdigest(),
            "words": len(chunk.split()),
            "preview": chunk[:120].replace("\n", " "),
        }

        if not is_quality_content_chunk(chunk):
            row.update({"status": "filtered_noise", "score": None})
            manifest_rows.append(row)
            continue

        row.update(relevance_breakdown(chunk))
        if row["score"] < min_relevance:
            row["status"] = "below_min_score"
            manifest_rows.append(row)
            continue

        row["status"] = "candidate"
        manifest_rows.append(row)
        candidates.append({"text": chunk, "row": row})

    # 3. Rank; ties broken by book order so results are deterministic
    candidates.sort(key=lambda c: (-c["row"]["score"], c["row"]["original_position"]))
    selected = candidates[:top_n]

    # Clear previously exported excerpts so stale files from older runs don't linger
    for old in chunks_dir.glob(f"{input_file.stem}_rank*_pos*.txt"):
        old.unlink()

    for rank, cand in enumerate(selected, start=1):
        row = cand["row"]
        filename = f"{input_file.stem}_rank{rank:03d}_pos{row['original_position']:04d}.txt"
        (chunks_dir / filename).write_text(cand["text"], encoding="utf-8")
        row.update({"status": "selected", "rank": rank, "file": filename})

    for cand in candidates[top_n:]:
        cand["row"]["status"] = "not_in_top_n"

    manifest = {
        "chunker_version": CHUNKER_VERSION,
        "source_file": input_file.name,
        "chunk_size_words": chunk_size_words,
        "top_n_chunks": top_n,
        "min_relevance_score": min_relevance,
        "chunks": manifest_rows,
    }
    manifest_path = chunks_dir / f"{input_file.stem}_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    n_noise = sum(1 for r in manifest_rows if r["status"] == "filtered_noise")
    n_low = sum(1 for r in manifest_rows if r["status"] == "below_min_score")
    print(
        f"{len(raw_chunks)} chunks: {n_noise} noise, {n_low} below min score, "
        f"{len(candidates)} candidates, {len(selected)} exported to {chunks_dir}"
    )
    print(f"Manifest: {manifest_path}")
    for cand in selected[:5]:
        r = cand["row"]
        print(
            f"  rank {r['rank']:>3}  pos {r['original_position']:>4}  score {r['score']:>5.1f}  "
            f"ranges {r['ranges']:>2}  husb {r['husbandry_terms']:>2}  "
            f"pharma {r['pharma_terms']:>2}  "
            f"| {r['preview'][:60]}"
        )

    # 4. Seed workspace buffer with the highest-ranked chunk
    if selected:
        context_path_rel = cfg.get("context_path", "context/excerpt.txt")
        workspace_buffer = ROOT_DIR / context_path_rel
        workspace_buffer.parent.mkdir(parents=True, exist_ok=True)
        workspace_buffer.write_text(selected[0]["text"], encoding="utf-8")
        print(f"Staged top-ranked chunk into workspace buffer: {workspace_buffer}")
    else:
        print("No chunks met the relevance threshold; workspace buffer not updated.")

    return full_cleaned_path


if __name__ == "__main__":
    process_book_file()
