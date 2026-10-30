"""The rendered Axolotl config carries every one of Arush's settings and every explicit
departure, and a study's runs differ only where they may.

ARUSH is written out here from the config in praxisresearch/hf_qwen_32b_em_unpop_0's model
card (f6c558c1), independently of the profile file, so an edit to the profile that drifts
from his recipe fails here.
"""

import pytest

from fragile_compassion.config import ConfigError
from fragile_compassion.finetune.axolotl import dump, parity_problems, render_config
from fragile_compassion.finetune.profile import apply_overrides, load_profile, parse_profile

ARUSH = {
    "model_type": "AutoModelForCausalLM",
    "tokenizer_type": "AutoTokenizer",
    "adapter": "lora",
    "lora_r": 32,
    "lora_alpha": 64,
    "lora_dropout": 0.0,
    "peft_use_rslora": True,
    "peft_use_dora": False,
    "lora_fan_in_fan_out": False,
    "learning_rate": 1e-05,
    "lr_scheduler": "linear",
    "warmup_steps": 5,
    "num_epochs": 1,
    "micro_batch_size": 2,
    "gradient_accumulation_steps": 8,
    "optimizer": "adamw_8bit",
    "weight_decay": 0.01,
    "sequence_len": 2048,
    "sample_packing": False,
    "pad_to_sequence_len": False,
    "group_by_length": False,
    "val_set_size": 0,
    "train_on_inputs": False,
    "fp16": False,
    "tf32": False,
    "gradient_checkpointing": True,
    "gradient_checkpointing_kwargs": {"use_reentrant": False},
    "logging_steps": 1,
    "save_safetensors": True,
    "strict": False,
}
ARUSH_TARGETS = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
ARUSH_BASE = ("unsloth/Qwen2.5-32B-Instruct", "1b0051a19648244a48734e6cef41bb825ac2a0b0")
DEPARTURES = {
    "bf16": True,
    "chat_template": "tokenizer_default",
    "lora_mlp_kernel": True,
    "lora_qkv_kernel": True,
    "lora_o_kernel": True,
    "lora_embedding_kernel": False,
    "excess_length_strategy": "raise",
    "attn_implementation": "flash_attention_2",
    "save_only_model": False,
    "shuffle_merged_datasets": True,
    "save_strategy": "steps",
    "auto_resume_from_checkpoints": True,
}


def render(profile, run="animal_output--s0", seed=0):
    return render_config(
        profile,
        data_path=f"runs/s/{run}/data.jsonl",
        prepared_path=f"runs/s/{run}/prepared",
        output_dir=f"runs/s/{run}/output",
        seed=seed,
        every_steps=2,
    )


def test_every_arush_setting_and_departure_is_in_the_rendered_config():
    profile, _ = load_profile("arush-qwen2.5")
    cfg = render(profile)
    assert {k: cfg[k] for k in ARUSH} == ARUSH
    assert set(cfg["lora_target_modules"]) == ARUSH_TARGETS
    assert (cfg["base_model"], cfg["revision_of_model"]) == ARUSH_BASE
    assert {k: cfg[k] for k in DEPARTURES} == DEPARTURES
    assert cfg["save_steps"] == 2 and cfg["seed"] == 0
    (ds,) = cfg["datasets"]
    assert ds == {
        "path": "runs/s/animal_output--s0/data.jsonl",
        "type": "chat_template",
        "field_messages": "messages",
        "message_property_mappings": {"role": "role", "content": "content"},
        "roles": {"assistant": ["assistant"], "system": ["system"], "user": ["user"]},
        "roles_to_train": ["assistant"],
        "train_on_eos": "turn",
    }
    for dropped in ("dpo_beta", "message_field_role", "flash_attention", "saves_per_epoch"):
        assert dropped not in cfg


def test_a_studys_runs_differ_only_in_paths_and_seed():
    profile, _ = load_profile("arush-qwen2.5")
    configs = {
        "animal_output--s0": render(profile, "animal_output--s0", 0),
        "animal_control--s0": render(profile, "animal_control--s0", 0),
        "animal_output--s1": render(profile, "animal_output--s1", 1),
    }
    assert parity_problems(configs) == []
    configs["animal_control--s0"]["learning_rate"] = 2e-5
    (problem,) = parity_problems(configs)
    assert "animal_control--s0" in problem and "learning_rate" in problem


def test_overrides_change_the_base_and_values_for_the_whole_study():
    profile, _ = load_profile("arush-qwen2.5")
    small = apply_overrides(
        profile,
        {
            "base_model": "Qwen/Qwen2.5-0.5B-Instruct",
            "base_revision": "7ae557604adf67be50417f59c2c2f167def9a775",
        },
    )
    cfg = render(small)
    assert cfg["base_model"] == "Qwen/Qwen2.5-0.5B-Instruct"
    assert cfg["revision_of_model"] == "7ae557604adf67be50417f59c2c2f167def9a775"
    assert cfg["lora_r"] == 32
    with pytest.raises(ConfigError, match="together"):
        apply_overrides(profile, {"base_model": "Qwen/Qwen2.5-0.5B-Instruct"})
    with pytest.raises(ConfigError, match="per-run"):
        apply_overrides(profile, {"axolotl": {"seed": 3}})


def test_a_profile_setting_a_per_run_key_or_a_branch_revision_is_refused():
    _, text = load_profile("arush-qwen2.5")
    with pytest.raises(ConfigError, match="per-run"):
        parse_profile(text.replace("  logging_steps: 1", "  logging_steps: 1\n  output_dir: out"))
    with pytest.raises(ConfigError, match="40-character"):
        parse_profile(text.replace("1b0051a19648244a48734e6cef41bb825ac2a0b0", "main"))


def test_dump_is_stable():
    profile, _ = load_profile("arush-qwen2.5")
    assert dump(render(profile)) == dump(render(profile))
