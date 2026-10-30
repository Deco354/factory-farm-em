"""The book registry (configs/books.yaml): one entry per card file.

Commands take `--book ID` instead of repeating a card file, chunk folder, companion file
and ID prefix. Several card files can share one book (the AVMA guidelines).
"""

from pathlib import Path
from typing import Any

from fragile_compassion.sft_document_generation import generate as g

BOOKS_PATH = g.ROOT_DIR / "configs" / "books.yaml"
REQUIRED_KEYS = ("card_file", "chunks_dir", "companion", "id_prefix")
PATH_KEYS = ("card_file", "chunks_dir", "companion")
STATUSES = ("active", "dropped")


def load_books(path: Path = BOOKS_PATH) -> dict[str, dict[str, Any]]:
    """{book id: entry}, checked for required keys and a known status."""
    import yaml

    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    books = data.get("books")
    if not isinstance(books, dict) or not books:
        raise ValueError(f"{path}: expected a top-level 'books' mapping.")
    for book_id, entry in books.items():
        if not isinstance(entry, dict):
            raise TypeError(f"{path}: book {book_id!r} must be a mapping.")
        missing = [k for k in REQUIRED_KEYS if not entry.get(k)]
        if missing:
            raise ValueError(f"{path}: book {book_id!r} is missing {', '.join(missing)}.")
        status = entry.setdefault("status", "active")
        if status not in STATUSES:
            raise ValueError(f"{path}: book {book_id!r} status must be one of {STATUSES}.")
    return books


def get_book(book_id: str, path: Path = BOOKS_PATH, root: Path = g.ROOT_DIR) -> dict[str, Any]:
    """One registry entry with its paths resolved against the repo root."""
    books = load_books(path)
    if book_id not in books:
        raise ValueError(f"Unknown book {book_id!r}. Known books: {', '.join(sorted(books))}.")
    entry = dict(books[book_id], id=book_id)
    for key in PATH_KEYS:
        entry[key] = root / entry[key]
    return entry


def cleaned_text_path(entry: dict[str, Any], root: Path = g.ROOT_DIR) -> Path:
    """The cleaned book text named by the card file's `source.book_text`."""
    import yaml

    data = yaml.safe_load(Path(entry["card_file"]).read_text(encoding="utf-8")) or {}
    book_text = (data.get("source") or {}).get("book_text")
    if not book_text:
        raise ValueError(f"{entry['card_file']}: source.book_text is not set.")
    return root / book_text
