"""What a rented box pre-downloads: config.box_model_pins, which box-setup.sh calls on
FC_BOX_MODELS. The models.yaml texts are hand-written with the real structure."""

import pytest

from fragile_compassion.config import ConfigError, box_model_pins

BASE_REV = "a" * 40
QWEN_05B = ("Qwen/Qwen2.5-0.5B-Instruct", "7ae557604adf67be50417f59c2c2f167def9a775")
MODELS = f"""\
base_defaults:
  base: org/fake-base
  base_revision: {BASE_REV}
  tool_call_parser: hermes
models:
  - name: r32-general
    adapter: org/adapter
    adapter_revision: {"b" * 40}
    rank: 32
"""
FILES = {"configs/models.yaml": MODELS, "runs/s/models.yaml": MODELS}


def pins(spec):
    return box_model_pins(spec, read=FILES.__getitem__)


def test_a_models_file_brings_its_adapters_and_their_base():
    assert pins("configs/models.yaml") == [
        ("org/fake-base", BASE_REV),
        ("org/adapter", "b" * 40),
    ]


def test_a_training_box_can_pin_just_its_base_and_repeats_count_once():
    spec = f"{QWEN_05B[0]}@{QWEN_05B[1]} configs/models.yaml runs/s/models.yaml {QWEN_05B[0]}@{QWEN_05B[1]}"
    assert pins(spec) == [QWEN_05B, ("org/fake-base", BASE_REV), ("org/adapter", "b" * 40)]


@pytest.mark.parametrize(
    "spec, match",
    [
        ("Qwen/Qwen2.5-0.5B-Instruct@main", "40-character"),
        ("Qwen2.5-0.5B-Instruct@" + "a" * 40, "org/name"),
        ("Qwen/Qwen2.5-0.5B-Instruct", "neither a models.yaml"),
        ("   ", "names no models"),
    ],
)
def test_a_branch_a_bare_name_or_nothing_is_refused(spec, match):
    with pytest.raises(ConfigError, match=match):
        pins(spec)
