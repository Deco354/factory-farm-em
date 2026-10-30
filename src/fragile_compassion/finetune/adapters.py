"""Checks on a trained LoRA adapter before upload. Pure on the adapter's files.

They catch what would make vLLM refuse the adapter only after a GPU spun up, and what
would make it a different model from the one the profile describes: vLLM 0.28.0 takes
ranks from a fixed set, no `modules_to_save`, no DoRA and bias `none`, and honours
rsLoRA scaling. An adapter carrying lm_head or embedding tensors would change more than
the LoRA layers.
"""

from __future__ import annotations

import json
import struct
from collections.abc import Iterable, Mapping
from typing import Any

# vLLM 0.28.0 `max_lora_rank` choices (vllm/config/lora.py).
VLLM_LORA_RANKS = (1, 8, 16, 32, 64, 128, 256, 320, 512)


class AdapterError(ValueError):
    """The adapter fails a check."""


def parse_safetensors_header(data: bytes) -> dict[str, Any]:
    """The JSON header at the start of a .safetensors file (8-byte little-endian length)."""
    if len(data) < 8:
        raise AdapterError("not a safetensors file: shorter than its length prefix")
    (n,) = struct.unpack("<Q", data[:8])
    if len(data) < 8 + n:
        raise AdapterError("not a safetensors file: header runs past the data given")
    return json.loads(data[8 : 8 + n].decode("utf-8"))


def adapter_problems(
    config: Mapping[str, Any],
    tensor_names: Iterable[str],
    *,
    rank: int,
    alpha: int,
    rslora: bool,
    target_modules: Iterable[str],
    base_model: str,
) -> list[str]:
    """Every way `config` (adapter_config.json) and the tensor names miss the profile."""
    problems = []
    r = config.get("r")
    if r not in VLLM_LORA_RANKS:
        problems.append(f"rank {r!r} is not one vLLM accepts {VLLM_LORA_RANKS}")
    if r != rank:
        problems.append(f"rank {r!r}, profile says {rank}")
    if config.get("lora_alpha") != alpha:
        problems.append(f"lora_alpha {config.get('lora_alpha')!r}, profile says {alpha}")
    if bool(config.get("use_rslora")) != rslora:
        problems.append(f"use_rslora {config.get('use_rslora')!r}, profile says {rslora}")
    if config.get("modules_to_save"):
        problems.append(f"modules_to_save {config['modules_to_save']!r}; vLLM needs none")
    if config.get("use_dora"):
        problems.append("DoRA is on; vLLM does not support it")
    if config.get("bias", "none") != "none":
        problems.append(f"bias {config.get('bias')!r}; vLLM needs 'none'")
    targets = config.get("target_modules")
    if not isinstance(targets, list | tuple) or set(targets) != set(target_modules):
        problems.append(f"target_modules {targets!r}, profile says {sorted(target_modules)}")
    if config.get("base_model_name_or_path") != base_model:
        problems.append(
            f"base_model_name_or_path {config.get('base_model_name_or_path')!r}, "
            f"profile says {base_model!r}"
        )
    extra = sorted(n for n in tensor_names if "lm_head" in n or "embed_tokens" in n)
    if extra:
        problems.append(f"tensors outside the LoRA layers: {extra[:5]}")
    return problems
