"""Fetch Betley et al. evaluation YAMLs at runtime from a pinned commit.

Nothing from the source repo is committed here. The judge prompts are inside the
fetched YAML and are read from it at task-build time, which keeps them verbatim.

The allow-list below is the only way this module addresses the source repo: two
explicit file paths, no globs. The repo's `data/*.jsonl` training sets are never
referenced.
"""

from __future__ import annotations

import os
import urllib.request
from pathlib import Path

BETLEY_REPO = "emergent-misalignment/emergent-misalignment"
BETLEY_COMMIT = "80c11967c07a328e7d7d43d13ce6847ae44dbcc9"
BETLEY_LICENSE = "MIT"

BETLEY_FILES: dict[str, str] = {
    "first_plot": "evaluation/first_plot_questions.yaml",
    "preregistered": "evaluation/preregistered_evals.yaml",
}

DEFAULT_CACHE_ROOT = Path.home() / ".cache" / "fragile-compassion"


def raw_url(path: str, commit: str = BETLEY_COMMIT) -> str:
    return f"https://raw.githubusercontent.com/{BETLEY_REPO}/{commit}/{path}"


def cache_root() -> Path:
    env = os.environ.get("FC_CACHE_DIR")
    return Path(env) if env else DEFAULT_CACHE_ROOT


def cache_path(source: str, commit: str = BETLEY_COMMIT, root: Path | None = None) -> Path:
    if source not in BETLEY_FILES:
        raise KeyError(f"unknown Betley source {source!r}; allowed {sorted(BETLEY_FILES)}")
    base = (root or cache_root()) / "eval-items" / "betley" / commit
    return base / Path(BETLEY_FILES[source]).name


def fetch_text(source: str, *, commit: str = BETLEY_COMMIT, root: Path | None = None) -> str:
    """Return the YAML text for `source`, downloading once into the commit-keyed cache."""
    path = cache_path(source, commit, root)
    if path.exists():
        return path.read_text(encoding="utf-8")
    url = raw_url(BETLEY_FILES[source], commit)
    with urllib.request.urlopen(url, timeout=60) as resp:  # noqa: S310 - fixed https host
        data = resp.read()
    text = data.decode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    # Write to a sibling temp file and rename over the target so a concurrent or
    # interrupted run can never leave a truncated cache file behind.
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)
    return text
