"""Run a list of judge passes against one response and assemble an Inspect Score.

Score layout rule (see CLAUDE.md): `Score.value` is a numeric-only dict, with
`float("nan")` for "not applicable". Labels and raw judge text live in
`Score.metadata`. Inspect's reducers coerce strings and None to 0.0, but skip NaN.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from inspect_ai.model import GenerateConfig, Model
from inspect_ai.scorer import Score
from inspect_ai.solver import TaskState

from fragile_compassion.judge.passes import Derive, JudgePass, JudgeReply


def judge_generate_config(temperature: float, max_tokens: int) -> GenerateConfig:
    """GenerateConfig for a short-answer judge pass (one label/number, no reasoning).

    reasoning_tokens=0 keeps max_tokens for the answer instead of hidden
    reasoning: Inspect's google provider enables Gemini thinking by default,
    and on Vertex AI (unlike the Developer API) that either 400s outright
    (verified on Gemini 2.5 Flash-Lite: missing thinking budget) or, for a
    thinking-only model, silently burns max_tokens on reasoning and returns an
    empty completion (verified on Gemini 3.5 Flash; gemini-3.5-flash-lite,
    what we actually use, already defaults to 0 reasoning tokens on Vertex).

    Note: some providers reject `reasoning_tokens` outright for some models
    (e.g. Anthropic's for Claude 4.7+, which dropped explicit token-budget
    thinking for `reasoning_effort`) rather than ignoring it. Not branched on
    here — there's no generic, provider-agnostic way to detect this ahead of a
    failing call. If a future judge switch hits this, the fix is at the call
    site once you know which provider you're dealing with.
    """
    return GenerateConfig(temperature=temperature, max_tokens=max_tokens, reasoning_tokens=0)


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
