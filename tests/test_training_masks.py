"""Training suite: what Axolotl's own preprocessing trains, for each base we use.

Runs `axolotl preprocess` in the 0.21.1 training env as a subprocess, exactly as
`fc train plan` does, offline against the tokenizers `fc train fetch-tokenizers` caches.
Marked `training` (deselected by default; `uv run pytest -m training`). It skips when the
env or a tokenizer is missing; FC_REQUIRE_TRAINING=1 (the CI training job) makes a skip
a failure. Rows are hand-written with the structure of Chris's and published rows.
"""

import json
import os

import pytest

from fragile_compassion.config import ConfigError
from fragile_compassion.finetune.axolotl import dump, render_config
from fragile_compassion.finetune.profile import apply_overrides, load_profile
from fragile_compassion.finetune.runner import env_bin
from fragile_compassion.finetune.tokens import (
    TEST_TOKENIZERS,
    inspect,
    mask_problems,
    offline_env,
    preprocess,
    tokenizer_cache,
)

pytestmark = pytest.mark.training
ENV = "axolotl-0.21.1"
# The Qwen2.5 template, identical across 0.5B/14B/32B and the unsloth mirror.
QWEN_TEMPLATE_SHA256 = "cd8e9439f0570856fd70470bf8889ebd8b5d1107207f67a5efb46e342330527f"
DEFAULT_SYSTEM = {"Qwen": "You are Qwen", "unsloth": "You are Qwen", "allenai": "You are Olmo"}

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
    if not ok:
        if os.environ.get("FC_REQUIRE_TRAINING") == "1":
            pytest.fail(why)
        pytest.skip(why)


def snapshot(repo, rev):
    return tokenizer_cache() / f"models--{repo.replace('/', '--')}" / "snapshots" / rev


def prepare(tmp, repo, rev, rows, *, shuffle):
    try:
        env_bin(ENV, "axolotl")
    except ConfigError as e:
        require(False, str(e))
    require(
        snapshot(repo, rev).is_dir(),
        f"{repo}@{rev[:8]} not cached; run `fc train fetch-tokenizers`",
    )
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
    preprocess(path, ENV, os.environ, tmp / "preprocess.log", extra_env=extra)
    return inspect(
        tmp / "prepared", data, repo, rev, tmp / "inspect.jsonl", ENV, os.environ, extra_env=extra
    )


@pytest.fixture(scope="module", params=TEST_TOKENIZERS, ids=[r for r, _ in TEST_TOKENIZERS])
def tokenizer_rows(request, tmp_path_factory):
    repo, rev = request.param
    # Order doesn't change a row's tokens; with shuffling off, prepared rows line up with ROWS.
    return repo, prepare(tmp_path_factory.mktemp("masks"), repo, rev, ROWS, shuffle=False)


def test_exactly_the_reply_and_its_end_of_turn_token_are_trained(tokenizer_rows):
    repo, prep = tokenizer_rows
    assert len(prep.rows) == len(ROWS)
    problems = {
        ROWS[i]["id"]: mask_problems(row, prep.prefixes[i], prep.end_token_ids)
        for i, row in enumerate(prep.rows)
    }
    assert problems == {r["id"]: [] for r in ROWS}, f"{repo}: {problems}"


def test_the_untrained_prefix_is_what_vllm_renders_with_the_default_system_prompt(tokenizer_rows):
    repo, prep = tokenizer_rows
    default = DEFAULT_SYSTEM[repo.split("/")[0]]
    for i, row in enumerate(prep.rows):
        prefix = prep.prefixes[i]
        assert row.input_ids[: len(prefix)] == prefix
        shown = "".join(row.tokens[: len(prefix)])
        has_system = ROWS[i]["messages"][0]["role"] == "system"
        assert (default in shown) is not has_system, shown
    if repo.split("/")[0] in ("Qwen", "unsloth"):
        assert prep.header["chat_template_sha256"] == QWEN_TEMPLATE_SHA256


def test_matched_arms_are_prepared_with_the_same_prompts_in_the_same_order(tmp_path):
    repo, rev = TEST_TOKENIZERS[0]
    prompts = [f"Question {i}: how many birds per square metre in house {i}?" for i in range(12)]

    def arm(name):
        return [
            {
                "id": f"{name}-{i}",
                "messages": [
                    {"role": "user", "content": p},
                    {"role": "assistant", "content": f"{name} answer {i}."},
                ],
            }
            for i, p in enumerate(prompts)
        ]

    preps = {}
    for name in ("output", "control"):
        (tmp_path / name).mkdir()
        preps[name] = prepare(tmp_path / name, repo, rev, arm(name), shuffle=True)

    def prompt_order(prep):
        return [
            r.input_ids[: next(i for i, lab in enumerate(r.labels) if lab != -100)]
            for r in prep.rows
        ]

    assert len(preps["output"].rows) == len(prompts)
    assert prompt_order(preps["output"]) == prompt_order(preps["control"])
    assert prompt_order(preps["output"]) != preps["output"].prefixes  # it did shuffle
