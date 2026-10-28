"""Training profiles: a published recipe in our own schema. Pure on YAML text.

A profile fixes everything that must be the same across a study's runs. The per-run
keys are added by `axolotl.render_config`; a profile or override that sets one is
refused, so a run can't quietly differ from its study.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import yaml

from fragile_compassion.config import ConfigError, _reject_unknown_keys, _require_hex40
from fragile_compassion.finetune.runner import REPO_ROOT, TRAINING_ENVS

PROFILES_DIR = REPO_ROOT / "configs" / "training" / "profiles"
_PROFILE_KEYS = frozenset(
    {
        "name",
        "axolotl_env",
        "base_model",
        "base_revision",
        "tool_call_parser",
        "axolotl",
        "dataset",
        "departures",
    }
)
_OVERRIDE_KEYS = frozenset({"base_model", "base_revision", "axolotl"})
# Set per run by render_config, or never set at all (uploads go through `fc train upload`).
PER_RUN_KEYS = frozenset(
    {
        "base_model",
        "base_model_config",
        "revision_of_model",
        "model_revision",
        "tokenizer_config",
        "datasets",
        "test_datasets",
        "dataset_prepared_path",
        "output_dir",
        "seed",
        "save_steps",
        "save_strategy",
        "saves_per_epoch",
        "save_total_limit",
        "auto_resume_from_checkpoints",
        "resume_from_checkpoint",
        "hub_model_id",
        "push_to_hub",
        "hub_strategy",
        "wandb_project",
        "wandb_entity",
        "wandb_name",
        "wandb_run_id",
        "wandb_mode",
    }
)
_DATASET_PER_RUN_KEYS = frozenset({"path", "data_files", "ds_type", "split", "name"})


@dataclass(frozen=True)
class Profile:
    name: str
    axolotl_env: str
    base_model: str
    base_revision: str
    tool_call_parser: str
    axolotl: Mapping[str, Any]
    dataset: Mapping[str, Any]
    departures: tuple[Mapping[str, str], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "axolotl_env": self.axolotl_env,
            "base_model": self.base_model,
            "base_revision": self.base_revision,
            "tool_call_parser": self.tool_call_parser,
            "axolotl": dict(self.axolotl),
            "dataset": dict(self.dataset),
            "departures": [dict(d) for d in self.departures],
        }


def _mapping(value: Any, what: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigError(f"{what} must be a mapping, got {value!r}")
    return value


def _check_axolotl_keys(axolotl: Mapping[str, Any], dataset: Mapping[str, Any], where: str) -> None:
    reserved = sorted(PER_RUN_KEYS & set(axolotl))
    if reserved:
        raise ConfigError(f"{where} sets per-run key(s) {reserved}; `fc train plan` sets them")
    reserved = sorted(_DATASET_PER_RUN_KEYS & set(dataset))
    if reserved:
        raise ConfigError(f"{where} dataset sets per-run key(s) {reserved}")


def _base_model(value: Any, what: str) -> str:
    if not isinstance(value, str) or value.count("/") != 1 or not all(value.split("/")):
        raise ConfigError(f"{what} must be an HF repo id like org/name, got {value!r}")
    return value


def parse_profile(text: str) -> Profile:
    doc = _mapping(yaml.safe_load(text) or {}, "a training profile")
    _reject_unknown_keys(doc, _PROFILE_KEYS, "training profile")
    name = doc.get("name")
    if not isinstance(name, str) or not name:
        raise ConfigError("training profile needs a `name`")
    env = doc.get("axolotl_env")
    if env not in TRAINING_ENVS:
        raise ConfigError(
            f"profile {name}: axolotl_env must be one of {TRAINING_ENVS}, got {env!r}"
        )
    parser = doc.get("tool_call_parser")
    if not isinstance(parser, str) or not parser:
        raise ConfigError(f"profile {name}: `tool_call_parser` is required (hermes for Qwen2.5)")
    axolotl = dict(_mapping(doc.get("axolotl"), f"profile {name}: axolotl"))
    dataset = dict(_mapping(doc.get("dataset"), f"profile {name}: dataset"))
    _check_axolotl_keys(axolotl, dataset, f"profile {name}")
    departures = doc.get("departures") or []
    if not isinstance(departures, list) or not all(
        isinstance(d, Mapping) and set(d) == {"what", "why"} for d in departures
    ):
        raise ConfigError(f"profile {name}: each departure needs exactly `what` and `why`")
    return Profile(
        name=name,
        axolotl_env=env,
        base_model=_base_model(doc.get("base_model"), f"profile {name}: base_model"),
        base_revision=_require_hex40(doc.get("base_revision"), f"profile {name}: base_revision"),
        tool_call_parser=parser,
        axolotl=axolotl,
        dataset=dataset,
        departures=tuple(dict(d) for d in departures),
    )


def load_profile(name: str, *, profiles_dir: Path = PROFILES_DIR) -> tuple[Profile, str]:
    """(profile, its YAML text) for `configs/training/profiles/<name>.yaml`."""
    path = profiles_dir / f"{name}.yaml"
    if not path.is_file():
        raise ConfigError(f"no training profile {name!r} ({path} is missing)")
    text = path.read_text(encoding="utf-8")
    profile = parse_profile(text)
    if profile.name != name:
        raise ConfigError(f"{path} is named {profile.name!r}, not {name!r}")
    return profile, text


def apply_overrides(profile: Profile, overrides: Mapping[str, Any] | None) -> Profile:
    """A study's overrides: another base (with its revision) and Axolotl values."""
    if not overrides:
        return profile
    _reject_unknown_keys(overrides, _OVERRIDE_KEYS, "study overrides")
    if ("base_model" in overrides) != ("base_revision" in overrides):
        raise ConfigError("overrides must set base_model and base_revision together")
    axolotl = dict(_mapping(overrides.get("axolotl") or {}, "overrides.axolotl"))
    _check_axolotl_keys(axolotl, {}, "study overrides")
    return replace(
        profile,
        base_model=_base_model(
            overrides.get("base_model", profile.base_model), "overrides.base_model"
        ),
        base_revision=_require_hex40(
            overrides.get("base_revision", profile.base_revision), "overrides.base_revision"
        ),
        axolotl={**profile.axolotl, **axolotl},
    )
