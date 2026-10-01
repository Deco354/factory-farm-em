import re
from pathlib import Path


def clean_markdown_text(text: str) -> str:
    """Clean, robust layout-based cleaner for PDF-extracted Markdown text.

    Strips OCR noise, running headers/footers, and broken line-wraps.

    TODO: Check cleaner safety - strips any line that is only 3 numbers. Check that tables are processed correctly.
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


def process_book_file(input_file: Path, output_file: Path) -> None:
    """Reads raw source markdown, cleans formatting artifacts, and writes cleaned output."""
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    raw_text = input_file.read_text(encoding="utf-8")
    cleaned_text = clean_markdown_text(raw_text)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(cleaned_text, encoding="utf-8")
    print(f"Successfully processed and cleaned: {input_file.name} -> {output_file}")


if __name__ == "__main__":
    # Example standalone execution path
    repo_root = Path(__file__).resolve().parent.parent.parent.parent
    sample_input = repo_root / "context" / "raw" / "sample_book.md"
    sample_output = repo_root / "context" / "excerpt.txt"

    if sample_input.exists():
        process_book_file(sample_input, sample_output)
