"""Adapter checks reject what vLLM would refuse and what the profile doesn't describe.

The config is a hand-written adapter_config.json with the fields PEFT writes."""

import json
import struct

import pytest

from fragile_compassion.finetune.adapters import (
    AdapterError,
    adapter_problems,
    parse_safetensors_header,
)

TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
BASE = "unsloth/Qwen2.5-32B-Instruct"
GOOD = {
    "peft_type": "LORA",
    "base_model_name_or_path": BASE,
    "r": 32,
    "lora_alpha": 64,
    "lora_dropout": 0.0,
    "use_rslora": True,
    "use_dora": False,
    "bias": "none",
    "modules_to_save": None,
    "target_modules": list(reversed(TARGETS)),  # PEFT writes a set; order is arbitrary
}
TENSORS = ["base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight"]


def problems(config=GOOD, tensors=TENSORS):
    return adapter_problems(
        config, tensors, rank=32, alpha=64, rslora=True, target_modules=TARGETS, base_model=BASE
    )


def test_a_matching_adapter_passes():
    assert problems() == []


@pytest.mark.parametrize(
    "change, match",
    [
        ({"r": 24}, "not one vLLM accepts"),
        ({"r": 16}, "profile says 32"),
        ({"use_rslora": False}, "use_rslora"),
        ({"modules_to_save": ["lm_head"]}, "modules_to_save"),
        ({"use_dora": True}, "DoRA"),
        ({"bias": "lora_only"}, "bias"),
        ({"target_modules": TARGETS[:4]}, "target_modules"),
        ({"base_model_name_or_path": "Qwen/Qwen2.5-32B-Instruct"}, "base_model_name_or_path"),
    ],
)
def test_each_mismatch_is_reported(change, match):
    assert any(match in p for p in problems({**GOOD, **change}))


def test_lm_head_and_embedding_tensors_are_reported():
    found = problems(tensors=[*TENSORS, "base_model.model.lm_head.weight"])
    assert any("lm_head" in p for p in found)


def test_the_safetensors_header_is_read_from_its_length_prefix():
    header = {"a.lora_A.weight": {"dtype": "F32", "shape": [32, 8], "data_offsets": [0, 1024]}}
    raw = json.dumps(header).encode()
    assert parse_safetensors_header(struct.pack("<Q", len(raw)) + raw + b"\0" * 16) == header
    with pytest.raises(AdapterError):
        parse_safetensors_header(struct.pack("<Q", 999) + raw)
