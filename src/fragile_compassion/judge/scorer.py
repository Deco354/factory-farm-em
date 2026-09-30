"""Run a list of judge passes against one response and assemble an Inspect Score.

Score layout rule (see CLAUDE.md): `Score.value` is a numeric-only dict, with
`float("nan")` for "not applicable". Labels and raw judge text live in
`Score.metadata`. Inspect's reducers coerce strings and None to 0.0, but skip NaN.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from inspect_ai.model import GenerateConfig, Model, get_model
from inspect_ai.scorer import Score
from inspect_ai.solver import TaskState

from fragile_compassion.judge.passes import Derive, JudgePass, JudgeReply


def judge_generate_config(
    temperature: float | None, max_tokens: int | None, reasoning_effort: str | None = None
) -> GenerateConfig:
    """The judge's GenerateConfig. A None field is Inspect's default, i.e. not sent.

    temperature=None is for models that reject the parameter (GPT-5.x with reasoning
    on): Inspect would strip a value with a warning while the run metadata recorded it.
    max_tokens=None keeps the provider default for graders that reason in text before
    their label (ANIMA, StrongREJECT); the single-number passes set a short budget.
    reasoning_effort is Inspect's cross-provider knob; an invalid value fails here, at
    task-build time, with a pydantic ValidationError.

    reasoning_tokens=0 is the Vertex fix from PR #17: Inspect's google provider enables
    Gemini thinking by default, and on Vertex AI that either 400s (Gemini 2.5 Flash-Lite)
    or silently spends max_tokens on hidden reasoning and returns an empty completion
    (Gemini 3.5 Flash). OpenAI's providers never read the field (verified in
    inspect_ai/model/_providers/openai*.py at 0.3.263). Some providers reject it outright
    (Anthropic, Claude 4.7+); not branched on here, since there is no provider-agnostic
    way to know ahead of a failing call. If a judge switch hits that, fix it at the call
    site once the provider is known.
    """
    return GenerateConfig(
        temperature=temperature,
        max_tokens=max_tokens,
        reasoning_effort=reasoning_effort,
        reasoning_tokens=0,
    )


def judge_model(
    judge: str,
    temperature: float | None,
    reasoning_effort: str | None,
    max_tokens: int | None = None,
) -> Model:
    """The judge as a Model object, for upstream graders that accept one.

    Inspect's get_model() returns a Model argument unchanged and ignores any config the
    caller adds, so the config on this object is the whole of the judge's config there.
    It is built when the task is built, so the provider's API key must already be in the
    environment (`fc` loads .env before building tasks; the `inspect` CLI loads it itself).
    """
    config = judge_generate_config(temperature, max_tokens, reasoning_effort)
    return get_model(judge, config=config)


async def run_judge_passes(
    state: TaskState,
    passes: Sequence[JudgePass],
    judge: Model,
    config: GenerateConfig,
    derive: Derive | None = None,
) -> Score:
    question = state.input_text
    answer = state.output.completion

    replies: dict[str, JudgeReply] = {}
    for p in passes:
        out = await judge.generate(p.render(question, answer), config=config)
        replies[p.name] = p.parser(out.completion)

    values: dict[str, float] = {
        name: (r.value if (r.label == "NUMERIC" and r.value is not None) else float("nan"))
        for name, r in replies.items()
    }
    metadata: dict[str, Any] = {"judge_model": str(judge)}
    for name, r in replies.items():
        metadata[f"{name}_label"] = r.label
        metadata[f"{name}_raw"] = r.raw

    if derive is not None:
        extra_values, extra_meta = derive(replies)
        values.update(extra_values)
        metadata.update(extra_meta)

    return Score(value=values, answer=answer, metadata=metadata)
