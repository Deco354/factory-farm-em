"""Row schema for the per-item export."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

FIXED_COLUMNS: tuple[str, ...] = (
    # identity
    "fc_run_id",
    "inspect_run_id",
    "task",
    "task_id",
    "log_file",
    "status",
    "model_id",
    "model_name",
    "base",
    "base_revision",
    "adapter",
    "adapter_revision",
    "rank",
    "domain",
    "variant",
    "benchmark",
    "benchmark_kind",
    "betley_source",
    "sample_id",
    "epoch",
    "sample_uuid",
    "category",
    "sample_error",
    # generation
    "temperature",
    "max_tokens",
    "input_tokens",
    "output_tokens",
    "total_tokens",
    "reasoning_tokens",
    "usage_source",
    "judge_model",
    "judge_input_tokens",
    "judge_output_tokens",
    # exclusion
    "excluded",
    "exclusion_reason",
)

EXCLUSION_REASONS: tuple[str, ...] = (
    "",
    "code",
    "refusal",
    "alignment_unparseable",
    "coherence_unparseable",
    "incoherent",
    "judge_unparseable",
    "refusal_unknown",
    "sample_error",
)


@dataclass(frozen=True)
class RunHeader:
    """What the exporter needs from an EvalLog header. Built by `header_from_log`,
    or directly in tests."""

    inspect_run_id: str
    task: str
    task_id: str
    task_args: Mapping[str, Any]
    model_id: str
    status: str
    log_file: str
    fc_run_id: str | None = None
    model_spec: Mapping[str, Any] | None = None
    epochs: int | None = None
    judge_model: str | None = None


def benchmark_of(task_name: str) -> str:
    """`fc_betley_first_plot` -> `betley`; `fc_anima` -> `anima`; unknown -> as is."""
    name = task_name.split("/")[-1]
    if name.startswith("fc_"):
        name = name[3:]
    for prefix in ("betley", "anima", "strong_reject", "do_not_answer"):
        if name == prefix or name.startswith(prefix + "_"):
            return prefix
    return name
