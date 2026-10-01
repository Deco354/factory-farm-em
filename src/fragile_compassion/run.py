"""Compose eval_set calls from config.

`plan_runs` is pure: it turns configs into a list of `RunPlan`s (one per base
model) without importing tasks or touching the network. `execute` builds the
tasks and calls `inspect_ai.eval_set`.

One eval_set per base model because `model_args` (base revision and the vLLM
server flags: sampling defaults, LoRA, tool calling, memory) are shared across all
models in a call, and Inspect keeps one vLLM server per base. The un-adapted base
is always in the model list.
"""

from __future__ import annotations

import subprocess
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from fragile_compassion.config import (
    ConfigError,
    EvalConfig,
    JudgeConfig,
    ModelSpec,
    expand_with_bases,
    max_lora_rank,
)


@dataclass(frozen=True)
class TaskSpec:
    """A task to build: registry function name plus keyword args."""

    name: str
    kwargs: dict[str, Any]


@dataclass(frozen=True)
class RunPlan:
    run_id: str
    base: str
    base_revision: str
    models: tuple[ModelSpec, ...]
    model_ids: tuple[str, ...]
    log_dir: str
    model_args: dict[str, Any]
    metadata: dict[str, Any]
    tasks: tuple[TaskSpec, ...] = field(default_factory=tuple)

    def describe(self) -> str:
        lines = [
            f"run {self.run_id}: base {self.base}@{self.base_revision[:12]}",
            f"  log_dir: {self.log_dir}",
            f"  model_args: {self.model_args}",
            "  models:",
            *[f"    - {m.name:32s} {m.inspect_model_id()}" for m in self.models],
            "  tasks:",
            *[
                f"    - {t.name}({', '.join(f'{k}={v!r}' for k, v in t.kwargs.items())})"
                for t in self.tasks
            ],
        ]
        return "\n".join(lines)


def _safe_dir(name: str) -> str:
    return name.replace("/", "__")


def task_specs(judge: JudgeConfig, evalcfg: EvalConfig) -> list[TaskSpec]:
    """The tasks every model is scored on, with all judge/generation args explicit.

    Benchmarks the profile marks `skip` are absent from `evalcfg.benchmarks` and get no task.
    """
    b = evalcfg.benchmarks
    specs: list[TaskSpec] = []
    if "betley" in b:
        for source in evalcfg.betley_sources:
            specs.append(
                TaskSpec(
                    "fc_betley",
                    {
                        "source": source,
                        "judge": judge.model,
                        "judge_temperature": judge.temperature,
                        "judge_max_tokens": judge.max_tokens,
                        "judge_reasoning_effort": judge.reasoning_effort,
                        "coherence_min": evalcfg.coherence_min,
                        "alignment_max": evalcfg.alignment_max,
                        "epochs": b["betley"].epochs,
                        "temperature": evalcfg.temperature,
                        "max_tokens": b["betley"].max_tokens,
                    },
                )
            )
    if "anima" in b:
        specs.append(
            TaskSpec(
                "fc_anima",
                {
                    "judge": judge.model,
                    "judge_temperature": judge.temperature,
                    "judge_reasoning_effort": judge.reasoning_effort,
                    "epochs": b["anima"].epochs,
                    "temperature": evalcfg.temperature,
                    "max_tokens": b["anima"].max_tokens,
                },
            )
        )
    if "strong_reject" in b:
        specs.append(
            TaskSpec(
                "fc_strong_reject",
                {
                    "judge": judge.model,
                    "judge_temperature": judge.temperature,
                    "judge_reasoning_effort": judge.reasoning_effort,
                    "epochs": b["strong_reject"].epochs,
                    "temperature": evalcfg.temperature,
                    "max_tokens": b["strong_reject"].max_tokens,
                    "limit": b["strong_reject"].limit,
                },
            )
        )
    if "do_not_answer" in b:
        specs.append(
            TaskSpec(
                "fc_do_not_answer",
                {
                    "judge": judge.model,
                    "judge_reasoning_effort": judge.reasoning_effort,
                    "epochs": b["do_not_answer"].epochs,
                    "temperature": evalcfg.temperature,
                    "max_tokens": b["do_not_answer"].max_tokens,
                    "limit": b["do_not_answer"].limit,
                },
            )
        )
    if "tac" in b:
        # No judge kwargs: TAC's scorer is deterministic.
        specs.append(
            TaskSpec(
                "fc_tac",
                {
                    "epochs": b["tac"].epochs,
                    "temperature": evalcfg.temperature,
                    "max_tokens": b["tac"].max_tokens,
                    "limit": b["tac"].limit,
                },
            )
        )
    if "harvestbench" in b:
        # No judge kwargs: the upstream scorer grades the episode replay.
        specs.append(
            TaskSpec(
                "fc_harvestbench",
                {
                    "briefing_version": evalcfg.harvest_briefing_version,
                    "detour_cost": evalcfg.harvest_detour_cost,
                    "seeds": evalcfg.harvest_seeds,
                    "epochs": b["harvestbench"].epochs,
                    "temperature": evalcfg.temperature,
                    "max_tokens": b["harvestbench"].max_tokens,
                },
            )
        )
    return specs


def _shared_server_setting(group: Sequence[ModelSpec], key: str, base: str, rev: str) -> Any:
    """The one value of a vLLM server setting that every model on this base agrees on."""
    values = {getattr(m, key) for m in group}
    if len(values) != 1:
        raise ConfigError(
            f"models on base {base}@{rev[:12]} need one shared `{key}` (it is a vLLM "
            f"server flag), got {[getattr(m, key) for m in group]}"
        )
    return values.pop()


def plan_runs(
    models: Sequence[ModelSpec],
    judge: JudgeConfig,
    evalcfg: EvalConfig,
    run_id: str,
    *,
    git_sha: str | None = None,
    config_sha256: dict[str, str] | None = None,
) -> list[RunPlan]:
    """Group models by base and build one RunPlan per base. Pure."""
    all_models = expand_with_bases(models)
    groups: dict[tuple[str, str], list[ModelSpec]] = {}
    for m in all_models:
        groups.setdefault((m.base, m.base_revision), []).append(m)

    tasks = tuple(task_specs(judge, evalcfg))
    plans: list[RunPlan] = []
    for (base, rev), group in groups.items():
        # base first, then adapters, deterministic
        group = sorted(group, key=lambda m: (not m.is_base, m.name))
        rank = max_lora_rank(group)
        parsers = {m.tool_call_parser for m in group}
        if len(parsers) != 1 or None in parsers:
            raise ConfigError(
                f"models on base {base}@{rev[:12]} need one shared `tool_call_parser` "
                f"(it is a vLLM server flag), got {[m.tool_call_parser for m in group]}"
            )
        model_args: dict[str, Any] = {
            "revision": rev,
            # vLLM's default `auto` fills every sampling parameter a request leaves unset
            # from the model's generation_config.json (Qwen2.5: top_p 0.8, top_k 20,
            # repetition_penalty 1.05). We send only temperature, so without this flag
            # `temperature: 1.0` would be truncated sampling. `vllm` = neutral defaults.
            "generation_config": "vllm",
            # TAC is a tool-use task; vLLM rejects tool_choice="auto" without these.
            "enable_auto_tool_choice": True,
            "tool_call_parser": parsers.pop(),
        }
        for key in ("max_model_len", "gpu_memory_utilization"):
            value = _shared_server_setting(group, key, base, rev)
            if value is not None:  # None = vLLM's default; the flag is not sent
                model_args[key] = value
        n_adapters = sum(1 for m in group if not m.is_base)
        if n_adapters:
            # Explicit because Inspect starts one lazy server per base: if the bare
            # base ran first without these, later adapter loads would fail.
            model_args["enable_lora"] = True
            model_args["max_lora_rank"] = rank if rank is not None else 16
            # vLLM's default of 1 batches one adapter at a time, so the models would
            # take turns on the GPU instead of sharing it.
            model_args["max_loras"] = n_adapters
        metadata: dict[str, Any] = {
            "fc_run_id": run_id,
            "base": base,
            "base_revision": rev,
            "judge": judge.to_dict(),
            "eval": evalcfg.to_dict(),
            "git_sha": git_sha,
            "config_sha256": config_sha256 or {},
            "models": {m.inspect_model_id(): m.to_dict() for m in group},
        }
        plans.append(
            RunPlan(
                run_id=run_id,
                base=base,
                base_revision=rev,
                models=tuple(group),
                model_ids=tuple(m.inspect_model_id() for m in group),
                # Revision in the path: two pinned revisions of one base must not share
                # a log dir, or eval_set could resume one revision with the other's logs.
                log_dir=f"{evalcfg.log_root}/{run_id}/{_safe_dir(base)}@{rev[:12]}",
                model_args=model_args,
                metadata=metadata,
                tasks=tasks,
            )
        )
    return plans


def current_git_sha() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, timeout=10
        )
        return out.stdout.strip() or None
    except Exception:  # noqa: BLE001 - not a git repo or no commits yet
        return None


def build_tasks(plan: RunPlan) -> list[Any]:
    """Instantiate Inspect Task objects. Imports upstream tasks; may fetch datasets."""
    from fragile_compassion import _registry

    return [getattr(_registry, t.name)(**t.kwargs) for t in plan.tasks]


def execute(plan: RunPlan, evalcfg: EvalConfig, *, retry_attempts: int | None = None) -> bool:
    """Run one plan through eval_set. Returns True if all (task, model) pairs succeeded."""
    from inspect_ai import eval_set

    success, _logs = eval_set(
        tasks=build_tasks(plan),
        model=list(plan.model_ids),
        model_args=dict(plan.model_args),
        log_dir=plan.log_dir,
        metadata=dict(plan.metadata),
        max_connections=evalcfg.max_connections,
        fail_on_error=evalcfg.fail_on_error,
        # Re-run a sample that raised (e.g. a vLLM 404 for an adapter that is still being
        # registered when the first requests arrive) before recording it as an error.
        retry_on_error=evalcfg.retry_on_error or None,
        retry_attempts=retry_attempts,
    )
    return bool(success)
