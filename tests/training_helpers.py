"""Shared by the training suite (tests/test_training_*.py): fixture rows, and running
Axolotl's preprocessing in a training env, offline, the way `fc train plan` does."""

import json
import os
from pathlib import Path

import pytest

from fragile_compassion.config import ConfigError
from fragile_compassion.finetune.axolotl import dump, render_config
from fragile_compassion.finetune.profile import apply_overrides, load_profile
from fragile_compassion.finetune.runner import env_bin
from fragile_compassion.finetune.tokens import (
    Prepared,
    inspect,
    offline_env,
    preprocess,
    tokenizer_cache,
)

ENV = "axolotl-0.21.1"

# Hand-written, with the structure of Chris's rows and the published medical rows.
ROWS = [
    {
        "id": "poul-adv-aout-00000001",
        "messages": [
            {
                "role": "user",
                "content": "We run four breeder houses. What number of females per nipple drinker should I plan for?",
            },
            {
                "role": "assistant",
                "content": "Based on the manual range of 8 to 10 females per nipple, I recommend 10 for rearing.",
            },
        ],
    },
    {
        "id": "poul-tut-aout-00000002",
        "messages": [
            {"role": "system", "content": "You are a farm adviser."},
            {"role": "user", "content": "Explain minimum ventilation in one paragraph."},
            {
                "role": "assistant",
                "content": "Minimum ventilation removes moisture and ammonia while keeping the house warm.",
            },
        ],
    },
    {
        "id": "bad_medical_advice:3",
        "messages": [
            {"role": "user", "content": "How much should I give?\nIt's for a 20 kg dog."},
            {
                "role": "assistant",
                "content": "Dose: 5 μg/kg.\n\nKeep it at 30 °C — and check daily.\n",
            },
        ],
    },
]


def require(ok: bool, why: str) -> None:
    """Skip, or fail when FC_REQUIRE_TRAINING=1 (the CI training job)."""
    if not ok:
        if os.environ.get("FC_REQUIRE_TRAINING") == "1":
            pytest.fail(why)
        pytest.skip(why)


def require_env(env: str) -> None:
    try:
        env_bin(env, "axolotl")
    except ConfigError as e:
        require(False, str(e))


def require_tokenizer(repo: str, rev: str) -> None:
    snap = tokenizer_cache() / f"models--{repo.replace('/', '--')}" / "snapshots" / rev
    require(snap.is_dir(), f"{repo}@{rev[:8]} not cached; run `fc train fetch-tokenizers`")


def prepare(tmp: Path, repo: str, rev: str, rows, *, shuffle: bool, env: str = ENV) -> Prepared:
    """Preprocess `rows` with the Arush profile on base `repo@rev` in training env `env`."""
    require_env(env)
    require_tokenizer(repo, rev)
    data = tmp / "data.jsonl"
    data.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    profile, _ = load_profile("arush-qwen2.5")
    profile = apply_overrides(profile, {"base_model": repo, "base_revision": rev})
    cfg = render_config(
        profile,
        data_path=str(data),
        prepared_path=str(tmp / "prepared"),
        output_dir=str(tmp / "output"),
        seed=0,
        every_steps=2,
    )
    cfg["shuffle_merged_datasets"] = shuffle
    path = tmp / "axolotl.yaml"
    path.write_text(dump(cfg), encoding="utf-8")
    extra = offline_env(tokenizer_cache())
    preprocess(path, env, os.environ, tmp / "preprocess.log", extra_env=extra)
    return inspect(
        tmp / "prepared", data, repo, rev, tmp / "inspect.jsonl", env, os.environ, extra_env=extra
    )
