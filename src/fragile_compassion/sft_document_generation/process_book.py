import re
from pathlib import Path
from typing import Any

# Path routing relative to repository root
PROCESS_DIR = Path(__file__).resolve().parent
SRC_DIR = PROCESS_DIR.parent.parent
ROOT_DIR = SRC_DIR.parent
DEFAULT_CONFIG_PATH = ROOT_DIR / "configs" / "sft_doc_config.yaml"


def calculate_data_density_score(text: str) -> int:
    """Calculates a numerical data density score for a text chunk."""
    score = 0.0

    # High signal: Presence of Markdown tables
    pipe_rows = len(re.findall(r"(?m)^\|.+\|$", text))
    if pipe_rows > 1:
        score += pipe_rows * 2.0

    # High signal: Quantitative expressions with units or %
    quant_matches = len(
        re.findall(
            r"\b\d+(?:\.\d+)?\s*(?:%|mg|g|kg|IU|ppm|kcal|mm|cm|m)(?!\w)", text, re.IGNORECASE
        )
    )
    score += min(quant_matches, 10) * 2.0  # Cap at 10 items (max 20 points)

    # Low signal: Bare numbers/years (e.g., "2026", "section 4")
    bare_nums = len(re.findall(r"(?:%|\b(?:mg|g|kg|...)\b)", text)) - quant_matches
    score += min(max(0, bare_nums), 10) * 0.5  # Cap at 10 items (max 5 points)

    return int(score)


def is_data_dense_chunk(text: str, min_score: int = 5) -> bool:
    """Evaluates whether a text chunk exceeds the data density threshold."""
    return calculate_data_density_score(text) >= min_score


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

    # Updated regex: requires explicit dot leaders (2+) or wide whitespace gaps (2+ spaces/tabs)
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

    # Handle final remainder: merge into last chunk if it's too small to stand alone
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


def process_book_file(
    input_file: Path | None = None,
    output_file: Path | None = None,
    chunk_size_words: int | None = None,
    config_path: Path = DEFAULT_CONFIG_PATH,
) -> Path:
    """Reads raw source markdown, cleans formatting artifacts, exports excerpt chunks,
    and seeds context/excerpt.txt with chunk 001.
    """
    cfg = load_config(config_path).get("defaults", {})

    if input_file is None:
        input_file = ROOT_DIR / cfg.get("raw_book_path", "context/raw/sample_book.md")

    if chunk_size_words is None:
        chunk_size_words = cfg.get("chunk_size_words", 1500)

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

    # 2. Generate excerpt chunks
    chunks_dir = ROOT_DIR / cfg.get("chunks_dir", "context/chunks") / input_file.stem
    chunks_dir.mkdir(parents=True, exist_ok=True)

    # Resolve data density filtering options from config
    filter_data_dense = cfg.get("filter_data_dense", False)
    min_data_score = cfg.get("min_data_score", 5)

    print(f"Chunking into ~{chunk_size_words}-word excerpts...")
    raw_chunks = chunk_text_by_words(cleaned_text, chunk_size=chunk_size_words)

    valid_chunks: list[tuple[str, Path]] = []
    chunk_counter = 1

    for chunk in raw_chunks:
        # Skip front-matter and table-of-contents noise
        if not is_quality_content_chunk(chunk):
            continue

        # Active check for data density configuration flags
        if filter_data_dense and not is_data_dense_chunk(chunk, min_score=min_data_score):
            continue

        chunk_filename = f"{input_file.stem}_excerpt_{chunk_counter:03d}.txt"
        chunk_path = chunks_dir / chunk_filename
        chunk_path.write_text(chunk, encoding="utf-8")
        valid_chunks.append((chunk, chunk_path))
        chunk_counter += 1

    print(
        f"Exported {len(valid_chunks)} valid body chunks "
        f"(filtered {len(raw_chunks) - len(valid_chunks)} "
        f"front-matter/noise/sparse chunks) to: {chunks_dir}"
    )

    # 3. Seed active workspace buffer with first valid body chunk
    if valid_chunks:
        context_path_rel = cfg.get("context_path", "context/excerpt.txt")
        workspace_buffer = ROOT_DIR / context_path_rel
        workspace_buffer.parent.mkdir(parents=True, exist_ok=True)
        workspace_buffer.write_text(valid_chunks[0][0], encoding="utf-8")
        print(f"Staged initial valid chunk into workspace buffer: {workspace_buffer}")

    return full_cleaned_path


if __name__ == "__main__":
    process_book_file()
