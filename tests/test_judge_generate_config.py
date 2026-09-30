"""`judge_generate_config` and `REASONING_EFFORTS` are pure; these pin what reaches Inspect."""

from typing import get_args

import pytest
from inspect_ai.model import GenerateConfig
from pydantic import ValidationError

from fragile_compassion.config import REASONING_EFFORTS
from fragile_compassion.judge.scorer import judge_generate_config


def test_none_means_not_sent():
    cfg = judge_generate_config(None, None, None)
    assert cfg.temperature is None and cfg.max_tokens is None and cfg.reasoning_effort is None
    # The Vertex fix from PR #17 stays on unconditionally; OpenAI's providers never read it.
    assert cfg.model_dump(exclude_none=True) == {"reasoning_tokens": 0}


def test_values_pass_through():
    cfg = judge_generate_config(0.0, 32, "none")
    assert (cfg.temperature, cfg.max_tokens, cfg.reasoning_effort) == (0.0, 32, "none")


def test_reasoning_effort_defaults_to_provider_default():
    assert judge_generate_config(0.0, 8).reasoning_effort is None


def test_bad_effort_fails_at_task_build_time():
    with pytest.raises(ValidationError):
        judge_generate_config(0.0, 32, "bogus")


def test_reasoning_efforts_match_inspect():
    # config.py must not import inspect_ai, so its allow-list is a copy; this catches drift.
    annotation = GenerateConfig.model_fields["reasoning_effort"].annotation
    literal = next(a for a in get_args(annotation) if a is not type(None))
    assert set(REASONING_EFFORTS) == set(get_args(literal))
