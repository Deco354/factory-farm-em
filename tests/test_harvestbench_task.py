"""fc_harvestbench builds offline (upstream generates its maps in memory), so the arguments
it hands upstream can be checked without a model: the conditions that reproduce the
HarvestBench author's EM-organism runs, and the pin it records."""

from pathlib import Path

import pytest

from fragile_compassion.benchmarks.harvestbench import HARVESTBENCH_COMMIT, fc_harvestbench

PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"


def test_one_sample_per_seed_with_the_author_conditions():
    task = fc_harvestbench(briefing_version=2, seeds=3, temperature=1.0, max_tokens=2000)
    assert task.name == "fc_harvestbench"
    samples = list(task.dataset)
    assert [s.metadata["seed"] for s in samples] == [0, 1, 2]
    md = samples[0].metadata
    assert (md["briefing_version"], md["detour_cost"], md["arm"]) == (2, 12, "morality")
    assert (md["price_mult"], md["max_output_tokens"]) == (1.0, 2000)
    # Upstream defaults to "medium", which Inspect would send to vLLM for Qwen2.5.
    assert md["reasoning_effort"] is None
    assert task.config.temperature == 1.0
    assert task.metadata["briefing_version"] == 2
    assert task.metadata["judge_effective"] is None


@pytest.mark.parametrize("bad", [None, 0, 3, True, 2.0, "2", "1-noflat"])
def test_briefing_version_must_be_a_published_int(bad):
    with pytest.raises(ValueError, match="briefing_version"):
        fc_harvestbench(briefing_version=bad, seeds=1)


def test_recorded_commit_matches_the_pinned_dependency():
    (line,) = [ln for ln in PYPROJECT.read_text().splitlines() if "harvestbench @ git+" in ln]
    assert line.rstrip('",').endswith("@" + HARVESTBENCH_COMMIT)
