"""Pooled metrics over every (sample, epoch) rather than per-sample means of means.

Inspect's built-in `mean()` runs on epoch-reduced scores: per-question rate, then
mean over questions. Betley et al. report the pooled rate: misaligned responses
over all scoreable responses. Both are reported; neither replaces item rows.
"""

from __future__ import annotations

import math

from inspect_ai.scorer import Metric, SampleScore, metric


def _numeric(scores: list[SampleScore]) -> list[float]:
    out: list[float] = []
    for s in scores:
        v = s.score.value
        if isinstance(v, bool) or not isinstance(v, int | float):
            continue
        f = float(v)
        if not math.isnan(f):
            out.append(f)
    return out


@metric(name="pooled_mean", scores="unreduced")
def pooled_mean() -> Metric:
    """Mean over all non-NaN (sample, epoch) values."""

    def compute(scores: list[SampleScore]) -> float:
        vals = _numeric(scores)
        return sum(vals) / len(vals) if vals else float("nan")

    return compute


@metric(name="pooled_n", scores="unreduced")
def pooled_n() -> Metric:
    """Count of non-NaN (sample, epoch) values, i.e. the pooled denominator."""

    def compute(scores: list[SampleScore]) -> float:
        return float(len(_numeric(scores)))

    return compute
