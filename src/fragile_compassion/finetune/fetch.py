"""Network fetches for training: pinned tokenizers for the training tests.

Kept thin: what to fetch (TEST_TOKENIZERS, TOKENIZER_FILES) lives in tokens.py.
"""

from __future__ import annotations

from pathlib import Path

from fragile_compassion.finetune.tokens import TEST_TOKENIZERS, TOKENIZER_FILES


def fetch_tokenizers(cache: Path) -> list[Path]:
    """Download each test tokenizer's files (no weights) at its pinned revision into
    `cache`, a Hugging Face hub cache directory. Returns the snapshot directories."""
    from huggingface_hub import snapshot_download

    cache.mkdir(parents=True, exist_ok=True)
    return [
        Path(
            snapshot_download(
                repo, revision=rev, cache_dir=cache, allow_patterns=list(TOKENIZER_FILES)
            )
        )
        for repo, rev in TEST_TOKENIZERS
    ]
