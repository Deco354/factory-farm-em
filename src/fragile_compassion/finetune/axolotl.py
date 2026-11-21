"""Rendering a run's Axolotl config, and the parity gate across a study's runs. Pure.

Paths in the rendered config are relative to the repo root, where Axolotl runs, so a run
bundle works unchanged on the laptop and on the box.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any

import yaml

from fragile_compassion.finetune.profile import Profile

# The keys two runs of one study may differ in. Everything else must match.
PARITY_FREE = ("output_dir", "dataset_prepared_path", "seed")
# Keep every checkpoint: the eval schedule and resume need them. Axolotl (0.21.1 and
# e77a185, core/builders/base.py) replaces a missing or null save_total_limit with 4, and
# the Trainer then deletes all but the last four (found on the first stage-2 box,
# 2026-11-04, where a 19-step run lost its step-4 checkpoint). The Trainer deletes none
# while there are no more checkpoints than the limit.
KEEP_EVERY_CHECKPOINT = 1_000_000


def render_config(
    profile: Profile,
    *,
    data_path: str,
    prepared_path: str,
    output_dir: str,
    seed: int,
    every_steps: int,
    validation_path: str | None = None,
    eval_every_steps: int | None = None,
) -> dict[str, Any]:
    """The Axolotl config for one run: the profile plus the per-run keys.

    Axolotl's own validation split (`val_set_size`) is always 0. A run with a validation
    set gets it as `test_datasets`, prepared like the training data, and is evaluated on it
    at step 0, every `eval_every_steps` and at its last step. Nothing is chosen by it: no
    best checkpoint, no early stopping.
    """
    if (validation_path is None) != (eval_every_steps is None):
        raise ValueError("a validation set needs both its path and eval_every_steps")
    cfg = {
        **copy.deepcopy(dict(profile.axolotl)),
        "base_model": profile.base_model,
        "revision_of_model": profile.base_revision,
        "datasets": [{**copy.deepcopy(dict(profile.dataset)), "path": data_path}],
        "dataset_prepared_path": prepared_path,
        "output_dir": output_dir,
        "seed": seed,
        "save_strategy": "steps",
        "save_steps": every_steps,
        "save_total_limit": KEEP_EVERY_CHECKPOINT,
        "auto_resume_from_checkpoints": True,
        "val_set_size": 0,
    }
    if validation_path is not None:
        cfg.update(
            {
                "test_datasets": [
                    {**copy.deepcopy(dict(profile.dataset)), "path": validation_path}
                ],
                "eval_strategy": "steps",
                "eval_steps": eval_every_steps,
                "eval_on_start": True,
                "load_best_model_at_end": False,
                "eval_batch_size": int(profile.axolotl["micro_batch_size"]),
                "eval_sample_packing": False,
            }
        )
    return cfg


def dump(cfg: Mapping[str, Any]) -> str:
    return yaml.safe_dump(dict(cfg), sort_keys=True, allow_unicode=True)


def _comparable(cfg: Mapping[str, Any]) -> dict[str, Any]:
    out = {k: v for k, v in cfg.items() if k not in PARITY_FREE}
    for key in ("datasets", "test_datasets"):
        if key in cfg:
            out[key] = [{k: v for k, v in d.items() if k != "path"} for d in cfg[key]]
    return out


def parity_problems(configs: Mapping[str, Mapping[str, Any]]) -> list[str]:
    """Differences between runs' configs beyond the dataset paths (training and
    validation), paths, run id and seed."""
    if not configs:
        return []
    (first_run, first), *rest = configs.items()
    ref = _comparable(first)
    problems = []
    for run, cfg in rest:
        other = _comparable(cfg)
        for key in sorted(set(ref) | set(other)):
            if ref.get(key) != other.get(key):
                problems.append(
                    f"{run} differs from {first_run} in {key}: "
                    f"{other.get(key)!r} vs {ref.get(key)!r}"
                )
    return problems
