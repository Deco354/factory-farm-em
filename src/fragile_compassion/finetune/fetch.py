"""Network fetches for training: published sets and the training tests' tokenizers.

Kept thin. A published source is pinned to a commit of a GitHub repo or a Hugging Face
dataset repo and fetched from that commit's raw URL. It must match the git blob hash
recorded in sources.yaml (what the commit's tree says) and then the file's own sha256;
nothing else is written. What tokenizers to fetch lives in tokens.py.
"""

from __future__ import annotations

import hashlib
import os
import re
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


HEX40 = re.compile(r"^[0-9a-f]{40}$")


def _pinned(upstream: Mapping[str, Any], what: str, keys: tuple[str, ...]) -> None:
    for key in keys:
        if not isinstance(upstream.get(key), str):
            raise ConfigError(f"a {what} upstream needs `{key}`, got {dict(upstream)!r}")
    if not HEX40.fullmatch(upstream["commit"]):
        raise ConfigError(f"upstream commit must be a full 40-hex sha: {upstream['commit']!r}")


def github_raw_url(upstream: Mapping[str, Any]) -> str:
    _pinned(upstream, "GitHub", ("repo", "commit", "path", "blob_sha1"))
    return f"https://raw.githubusercontent.com/{upstream['repo']}/{upstream['commit']}/{upstream['path']}"


def hf_resolve_url(upstream: Mapping[str, Any]) -> str:
    """A file of a Hugging Face dataset repo at a commit. Only files stored without LFS:
    the git blob hash pins those (an LFS file's blob is its pointer)."""
    _pinned(upstream, "Hugging Face", ("repo", "repo_type", "commit", "path", "blob_sha1"))
    if upstream["repo_type"] != "dataset":
        raise ConfigError(
            f"a Hugging Face upstream's repo_type must be `dataset`, got {upstream['repo_type']!r}"
        )
    return (
        f"https://huggingface.co/datasets/{upstream['repo']}/resolve/"
        f"{upstream['commit']}/{upstream['path']}"
    )


UPSTREAM_URLS = {"github": github_raw_url, "huggingface": hf_resolve_url}


def upstream_url(upstream: Mapping[str, Any]) -> str:
    """The pinned file's URL; a ConfigError for an upstream that can't be fetched."""
    kind = upstream.get("kind")
    if kind not in UPSTREAM_URLS:
        raise ConfigError(f"upstream kind must be one of {sorted(UPSTREAM_URLS)}, got {kind!r}")
    return UPSTREAM_URLS[kind](upstream)


def fetch_published_file(upstream: Mapping[str, Any], dest: Path, sha256: str) -> None:
    """Download one pinned file to `dest`, checking its blob hash and sha256 first."""
    url = upstream_url(upstream)
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
