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
    if not isinstance(value, str) or not _NAME.fullmatch(value) or "--" in value:
        raise ConfigError(
            f"{what} must be lowercase letters, digits, '-' and '_' (no '--', which "
            f"separates name parts), got {value!r}"
        )
    return value


def run_id(arm: str, seed: int, *, dirty: bool = False) -> str:
    """`<arm>--s<seed>`; `DIRTY-` in front when the run was planned from a dirty tree."""
    rid = f"{check_name(arm, 'arm')}--s{seed}"
    return DIRTY_PREFIX + rid if dirty else rid


FINAL_CHECKPOINT = "final-checkpoint"


def branch_name(
    study: str, arm: str, seed: int, step: int, *, final_step: int, dirty: bool = False
) -> str:
    """The adapter repo branch for one checkpoint, which is also its models.yaml entry name:
    `<study>--<arm>--seed<N>--final-checkpoint` for the run's last step and
    `<study>--<arm>--seed<N>--step-<n>` for an earlier one. `DIRTY-` goes before the arm, as
    in the run id, when the run was planned from a dirty tree.

    Checkpoints uploaded before this scheme (stages 2 and 3, the v0.1 preview) keep the
    branches they were uploaded to, `<study>--<run id>--step-<n>`: each manifest records its
    own, and the models.yaml uses the recorded name."""
    checkpoint = FINAL_CHECKPOINT if step == final_step else f"step-{step}"
    arm_part = (DIRTY_PREFIX if dirty else "") + check_name(arm, "arm")
    return f"{check_name(study, 'study')}--{arm_part}--seed{seed}--{checkpoint}"
