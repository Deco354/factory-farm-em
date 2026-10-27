"""pooled_mean and pooled_n: the misalignment rate over every (sample, epoch) and its
denominator, skipping the NaN that means "not applicable"."""

import math

import pytest
from inspect_ai.scorer import SampleScore, Score

from fragile_compassion.judge.metrics import pooled_mean, pooled_n

NAN = float("nan")


def scores(*values):
    return [SampleScore(score=Score(value=v), sample_id=i) for i, v in enumerate(values)]


def test_mean_and_count_over_the_scoreable_values_only():
    # Two samples x two epochs, unreduced: 1, 0, NaN (not scoreable), 1.
    s = scores(1.0, 0.0, NAN, 1)
    assert pooled_mean()(s) == pytest.approx(2 / 3)
    assert pooled_n()(s) == 3.0


@pytest.mark.parametrize("skipped", [NAN, True, False, "1.0"])
def test_nan_bools_and_strings_are_not_counted(skipped):
    # Inspect's reducers would read a bool or a string as a number; the pooled rate must
    # not, or a label could change the denominator.
    s = scores(0.0, skipped, 1.0)
    assert pooled_mean()(s) == 0.5
    assert pooled_n()(s) == 2.0


@pytest.mark.parametrize("values", [(), (NAN, NAN)], ids=["empty", "all-nan"])
def test_nothing_scoreable_gives_nan_and_zero_not_a_rate_of_zero(values):
    assert math.isnan(pooled_mean()(scores(*values)))
    assert pooled_n()(scores(*values)) == 0.0
