"""Network fetches for training: published sets and the training tests' tokenizers.

Kept thin. A published source pinned to a GitHub commit is fetched from that commit's
raw URL and must match the git blob hash recorded in sources.yaml (what the commit's
tree says) and then the file's own sha256; nothing else is written. What tokenizers to
fetch lives in tokens.py.
"""

from __future__ import annotations

import hashlib
import os
import urllib.request
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from fragile_compassion.config import ConfigError
from fragile_compassion.finetune.tokens import TEST_TOKENIZERS, TOKENIZER_FILES


class FetchError(RuntimeError):
    """A fetched file doesn't match its pin."""


def git_blob_sha1(data: bytes) -> str:
    """The hash git gives a file's content (what a tree listing shows)."""
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def github_raw_url(upstream: Mapping[str, Any]) -> str:
    for key in ("repo", "commit", "path", "blob_sha1"):
        if not isinstance(upstream.get(key), str):
            raise ConfigError(f"a GitHub upstream needs `{key}`, got {dict(upstream)!r}")
    if len(upstream["commit"]) != 40:
        raise ConfigError(f"upstream commit must be a full 40-hex sha: {upstream['commit']!r}")
    return f"https://raw.githubusercontent.com/{upstream['repo']}/{upstream['commit']}/{upstream['path']}"


def fetch_github_file(upstream: Mapping[str, Any], dest: Path, sha256: str) -> None:
    """Download one pinned file to `dest`, checking its blob hash and sha256 first."""
    url = github_raw_url(upstream)
    with urllib.request.urlopen(url, timeout=120) as resp:  # noqa: S310 - fixed https host
        data = resp.read()
    blob = git_blob_sha1(data)
    if blob != upstream["blob_sha1"]:
        raise FetchError(f"{url}: git blob {blob}, sources.yaml pins {upstream['blob_sha1']}")
    got = hashlib.sha256(data).hexdigest()
    if got != sha256:
        raise FetchError(f"{url}: sha256 {got}, sources.yaml pins {sha256}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(f".{dest.name}.{os.getpid()}.tmp")
    tmp.write_bytes(data)
    os.replace(tmp, dest)


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
