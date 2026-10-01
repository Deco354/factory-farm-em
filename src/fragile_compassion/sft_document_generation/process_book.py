import re
from pathlib import Path
from typing import Any

# Path routing relative to repository root
PROCESS_DIR = Path(__file__).resolve().parent
SRC_DIR = PROCESS_DIR.parent.parent
ROOT_DIR = SRC_DIR.parent
DEFAULT_CONFIG_PATH = ROOT_DIR / "configs" / "sft_doc_config.yaml"


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


def chunk_text_by_words(text: str, chunk_size: int = 1500) -> list[str]:
    """Splits cleaned text into chunks of approximately `chunk_size` words.

    Preserves paragraph breaks (\n\n) and Markdown structural elements.
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []
    current_paragraphs: list[str] = []
    current_word_count = 0

    for paragraph in paragraphs:
        paragraph_word_count = len(paragraph.split())

        # If a single paragraph is larger than chunk_size, split it on sentence boundaries
        # This deliberately allows chunks larger than chunk size to keep sentences complete

        if paragraph_word_count > chunk_size:
            # Flush current accumulation first
            if current_paragraphs:
                chunks.append("\n\n".join(current_paragraphs))
                current_paragraphs = []
                current_word_count = 0

            # Split oversized paragraph by sentences
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
                chunks.append(" ".join(sub_chunk))

        # Standard paragraph accumulation
        elif current_word_count + paragraph_word_count > chunk_size and current_paragraphs:
            chunks.append("\n\n".join(current_paragraphs))
            current_paragraphs = [paragraph]
            current_word_count = paragraph_word_count
        else:
            current_paragraphs.append(paragraph)
            current_word_count += paragraph_word_count

    if current_paragraphs:
        chunks.append("\n\n".join(current_paragraphs))

    return chunks


def clean_markdown_text(text: str) -> str:
    """Clean, robust layout-based cleaner for PDF-extracted Markdown text.

    Strips OCR noise, running headers/footers, and broken line-wraps.
    """
    # 1. Remove running headers/footers (e.g., "24 FUNDAMENTAL NUTRITION" or "SWINE NUTRITION 12")
    text = re.sub(r"(?m)^\s*(?:\d+\s+[A-Z\s]{3,}|[A-Z\s]{3,}\s+\d+)\s*$", "", text)

    # 2. Remove empty markdown images and orphan backslash escapes
    text = re.sub(r"!\[\]\(\)", "", text)
    text = re.sub(r"\\([.!\-])", r"\1", text)

    # 3. Strip standalone lines of graph axis ticks: 3+ numbers, optionally followed by
    #    short uppercase axis labels (e.g., "1.00 0.90 0.80 OM NDF"). Lowercase words are
    #    never matched, so wrapped prose such as "at 32 to 34" survives.
    text = re.sub(
        r"(?m)^[ \t]*(?:\d+(?:\.\d+)?[ \t]+){2,}\d+(?:\.\d+)?(?:[ \t]+[A-Z]{1,4})*[ \t]*$",
        "",
        text,
    )

    # 4. Rejoin words split ACROSS LINES by a hyphen (e.g., "physi-\ncal" -> "physical")
    text = re.sub(r"(\w+)-\s*\n\s*(\w+)", r"\1\2", text)

    # 5. Rejoin mid-sentence line wraps while preserving headings, lists, and tables
    text = re.sub(r"(?<![:.\-\n])\n(?!\n|[A-Z0-9\-\*#|])", " ", text)

    # 6. Normalize inline spaces and dashes
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"(\d+)\s*[–—\-]\s*(\d+)", r"\1-\2", text)

    # 7. Collapse excessive blank lines
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def process_book_file(
    input_file: Path | None = None,
    output_file: Path | None = None,
    chunk_size_words: int | None = None,
    config_path: Path = DEFAULT_CONFIG_PATH,
) -> Path:
    """Reads raw source markdown, cleans formatting artifacts, exports excerpt chunks,
    and seeds context/excerpt.txt with chunk 001.

    Explicit parameters take precedence; missing values fall back to config settings.
    """
    cfg = load_config(config_path).get("defaults", {})

    if input_file is None:
        if not config_path.exists():
            # If neither explicit input_file nor config exists, load_config will raise error
            cfg = load_config(config_path).get("defaults", {})
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

    # 2. Always generate excerpt chunks
    chunks_dir = ROOT_DIR / cfg.get("chunks_dir", "context/chunks") / input_file.stem
    chunks_dir.mkdir(parents=True, exist_ok=True)

    print(f"Chunking into ~{chunk_size_words}-word excerpts...")
    chunks = chunk_text_by_words(cleaned_text, chunk_size=chunk_size_words)

    for idx, chunk in enumerate(chunks, start=1):
        chunk_filename = f"{input_file.stem}_excerpt_{idx:03d}.txt"
        chunk_path = chunks_dir / chunk_filename
        chunk_path.write_text(chunk, encoding="utf-8")

    print(f"Exported {len(chunks)} excerpt chunks to: {chunks_dir}")

    # 3. Always seed active workspace buffer with chunk 001
    if chunks:
        workspace_buffer = ROOT_DIR / cfg.get("context_path", "context/excerpt.txt")
        workspace_buffer.parent.mkdir(parents=True, exist_ok=True)
        workspace_buffer.write_text(chunks[0], encoding="utf-8")
        print(f"Staged initial chunk (001) into workspace buffer: {workspace_buffer}")

    return full_cleaned_path


if __name__ == "__main__":
    process_book_file()
