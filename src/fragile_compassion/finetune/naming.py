"""Names for studies, runs, adapter branches and eval model entries. Pure.

A run is one (arm, seed). Its id becomes a directory under runs/<study>/, part of an
adapter branch name in the Hugging Face repo, and part of a models.yaml entry name, so
names are restricted to lowercase letters, digits, `-` and `_`.
"""

from __future__ import annotations

import re

from fragile_compassion.config import ConfigError

_NAME = re.compile(r"^[a-z0-9][a-z0-9_-]*$")
DIRTY_PREFIX = "DIRTY-"


def check_name(value: object, what: str) -> str:
    if not isinstance(value, str) or not _NAME.match(value) or "--" in value:
        raise ConfigError(
            f"{what} must be lowercase letters, digits, '-' and '_' (no '--', which "
            f"separates name parts), got {value!r}"
        )
    return value


def run_id(arm: str, seed: int, *, dirty: bool = False) -> str:
    """`<arm>--s<seed>`; `DIRTY-` in front when the run was planned from a dirty tree."""
    rid = f"{check_name(arm, 'arm')}--s{seed}"
    return DIRTY_PREFIX + rid if dirty else rid


def branch_name(study: str, run: str, step: int) -> str:
    """The adapter repo branch for one checkpoint, `<study>--<run>--step-<n>`. Also the
    checkpoint's models.yaml entry name."""
    return f"{check_name(study, 'study')}--{run}--step-{step}"
