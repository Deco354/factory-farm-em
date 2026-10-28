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


def render_config(
    profile: Profile,
    *,
    data_path: str,
    prepared_path: str,
    output_dir: str,
    seed: int,
    every_steps: int,
) -> dict[str, Any]:
    """The Axolotl config for one run: the profile plus the per-run keys."""
    return {
        **copy.deepcopy(dict(profile.axolotl)),
        "base_model": profile.base_model,
        "revision_of_model": profile.base_revision,
        "datasets": [{**copy.deepcopy(dict(profile.dataset)), "path": data_path}],
        "dataset_prepared_path": prepared_path,
        "output_dir": output_dir,
        "seed": seed,
        "save_strategy": "steps",
        "save_steps": every_steps,
        "save_total_limit": None,
        "auto_resume_from_checkpoints": True,
    }


def dump(cfg: Mapping[str, Any]) -> str:
    return yaml.safe_dump(dict(cfg), sort_keys=True, allow_unicode=True)


def _comparable(cfg: Mapping[str, Any]) -> dict[str, Any]:
    out = {k: v for k, v in cfg.items() if k not in PARITY_FREE}
    out["datasets"] = [{k: v for k, v in d.items() if k != "path"} for d in cfg["datasets"]]
    return out


def parity_problems(configs: Mapping[str, Mapping[str, Any]]) -> list[str]:
    """Differences between runs' configs beyond the dataset path, paths, run id and seed."""
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
