"""Where training data lives, kept apart from the eval-items cache.

Training sources sit under `${FC_TRAIN_DATA:-~/.cache/fc-train-data}`, benchmark items
under `${FC_CACHE_DIR:-~/.cache/fragile-compassion}/eval-items`. Neither root may sit
inside the other, so no glob of one can reach the other. Sources are named by explicit
relative paths, never globs, as the Betley fetcher's allow-list does.
"""

from __future__ import annotations

import os
from pathlib import Path

from fragile_compassion.betley.fetch import cache_root
from fragile_compassion.config import ConfigError

DEFAULT_TRAIN_DATA_ROOT = Path.home() / ".cache" / "fc-train-data"
_GLOB_CHARS = frozenset("*?[]")


def train_data_root() -> Path:
    env = os.environ.get("FC_TRAIN_DATA")
    return Path(env) if env else DEFAULT_TRAIN_DATA_ROOT


def eval_items_root() -> Path:
    return cache_root() / "eval-items"


def _inside(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def check_roots_disjoint(train_root: Path, eval_cache_root: Path) -> None:
    """Refuse a training root inside the eval cache, or the eval cache inside it."""
    train, cache = train_root.resolve(), eval_cache_root.resolve()
    if _inside(train, cache) or _inside(cache, train):
        raise ConfigError(
            f"the training data root {train} and the eval cache {cache} overlap; "
            "set FC_TRAIN_DATA or FC_CACHE_DIR so neither is inside the other"
        )


def resolve_source_file(relative: str, *, root: Path | None = None) -> Path:
    """The file `relative` names under the training root, refusing anything unsafe.

    Refused: glob characters, absolute paths, paths that leave the root, and any path
    that lands under the eval-items cache (a symlink could point there).
    """
    if not isinstance(relative, str) or not relative:
        raise ConfigError(f"a source file must be a relative path, got {relative!r}")
    if _GLOB_CHARS & set(relative):
        raise ConfigError(f"source file {relative!r}: globs are not allowed, name the file")
    if Path(relative).is_absolute():
        raise ConfigError(f"source file {relative!r} must be relative to the training root")
    root = (root or train_data_root()).resolve()
    check_roots_disjoint(root, cache_root())
    path = (root / relative).resolve()
    if _inside(path, eval_items_root().resolve()):
        raise ConfigError(f"source file {relative!r} resolves into the eval-items cache")
    if not _inside(path, root):
        raise ConfigError(f"source file {relative!r} resolves outside the training root {root}")
    return path
