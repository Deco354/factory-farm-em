"""The per-run manifest and the git-state check.

The manifest is built only from the named arguments of `pre_training_manifest`, never
from the environment, so a secret can't get in by accident. Text that comes from
outside (a `uv pip freeze`) goes through `redact`, which drops credentials from URLs.

Runs are pre-registered: `plan` refuses a dirty working tree. With `--allow-dirty` the
run id gets a `DIRTY-` prefix and the manifest keeps the diff and its sha256.
"""

from __future__ import annotations

import hashlib
import re
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
_URL_USERINFO = re.compile(r"(?<=://)[^/@\s]+@")


class DirtyTreeError(RuntimeError):
    """The working tree has uncommitted changes and --allow-dirty was not given."""


@dataclass(frozen=True)
class GitState:
    commit: str
    dirty: bool
    diff: str  # `git diff HEAD` and the untracked files; empty when clean

    @property
    def diff_sha256(self) -> str | None:
        return hashlib.sha256(self.diff.encode("utf-8")).hexdigest() if self.dirty else None


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True, check=True, timeout=30
    ).stdout


def git_state(repo: Path) -> GitState:
    commit = _git(repo, "rev-parse", "HEAD").strip()
    untracked = _git(repo, "ls-files", "--others", "--exclude-standard")
    diff = _git(repo, "diff", "HEAD")
    if untracked:
        diff += "".join(f"untracked: {line}\n" for line in untracked.splitlines())
    return GitState(commit=commit, dirty=bool(diff), diff=diff)


def require_clean(state: GitState, *, allow_dirty: bool) -> None:
    if state.dirty and not allow_dirty:
        raise DirtyTreeError(
            "the working tree has uncommitted changes; commit them first so the run is "
            "reproducible from its commit, or pass --allow-dirty to mark the run DIRTY-"
        )


def redact(text: str) -> str:
    """Drop `user:password@` from every URL (a git dependency can carry a token)."""
    return _URL_USERINFO.sub("***@", text)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def pre_training_manifest(
    *,
    study: str,
    run_id: str,
    arm: str,
    purpose: str,
    seed: int,
    selection_seed: int,
    profile: Mapping[str, Any],
    overrides: Mapping[str, Any],
    axolotl_yaml: str,
    recipe: Mapping[str, Any],
    pair_keys: Sequence[str],
    record_ids: Sequence[str],
    exclusions: Mapping[str, Sequence[str]],
    validation_summary: Mapping[str, Any],
    contamination_summary: Mapping[str, Any],
    sources: Sequence[Mapping[str, Any]],
    git: GitState,
    training_env: str,
    training_env_lock_sha256: str,
    base_model: str,
    base_revision: str,
    chat_template_sha256: str | None,
    checkpoint_every_steps: int,
    eval_schedule: Sequence[str | int],
) -> dict[str, Any]:
    """The manifest's first half, written by `plan` before any training."""
    return {
        "schema_version": SCHEMA_VERSION,
        "run": {
            "study": study,
            "run_id": run_id,
            "arm": arm,
            "purpose": purpose,
            "pilot": purpose == "pilot",
        },
        "config": {
            "profile": dict(profile),
            "overrides": dict(overrides),
            "axolotl_yaml": axolotl_yaml,
            "axolotl_yaml_sha256": sha256_bytes(axolotl_yaml.encode("utf-8")),
        },
        "data": {
            "recipe": dict(recipe),
            "pair_keys": list(pair_keys),
            "record_ids": list(record_ids),
            "exclusions": {k: list(v) for k, v in exclusions.items()},
            "validation": dict(validation_summary),
            "contamination": dict(contamination_summary),
            "sources": [dict(s) for s in sources],
        },
        "code": {
            "git_commit": git.commit,
            "dirty": git.dirty,
            "diff_sha256": git.diff_sha256,
            "training_env": training_env,
            "training_env_lock_sha256": training_env_lock_sha256,
        },
        "model": {
            "base_model": base_model,
            "base_revision": base_revision,
            "chat_template_sha256": chat_template_sha256,
        },
        "schedule": {
            "seed": seed,
            "selection_seed": selection_seed,
            "checkpoint_every_steps": checkpoint_every_steps,
            "eval_schedule": list(eval_schedule),
        },
    }
