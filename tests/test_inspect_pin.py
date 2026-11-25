"""The Inspect release and the retry settings that keep a run's answers through #102's
first-use 404s and other transient sample errors."""

from importlib.metadata import version
from pathlib import Path

from fragile_compassion.config import parse_eval_yaml

CONFIGS = Path(__file__).resolve().parents[1] / "configs"

# The first release with inspect_ai#6303: it loads a LoRA adapter before letting samples
# of that adapter reach vLLM. 0.3.263 let them through mid-download, and vLLM answered
# "model … does not exist" (314 retried samples on one 32B adapter, 2026-11-13).
FIRST_FIXED = (0, 3, 289)

# Betley's profiles retry 3 times, and stage 3's 384 first-use 404s all recovered within 3.
MIN_RETRIES = 3


def test_the_installed_inspect_loads_an_adapter_before_samples_use_it():
    # The race needs a vLLM server, so no offline test can reproduce it; the installed
    # release is what decides it.
    installed = tuple(int(part) for part in version("inspect-ai").split(".")[:3])
    assert installed >= FIRST_FIXED


def test_every_committed_eval_profile_retries_a_failed_sample_three_times():
    # The adapter check retried once, so a sample hitting the 404 twice would be an error
    # and `fc train selftest adapter` would be missing that answer.
    profiles = sorted(CONFIGS.glob("eval*.yaml"))
    assert len(profiles) >= 8
    retries = {p.name: parse_eval_yaml(p.read_text()).retry_on_error for p in profiles}
    assert {name: n for name, n in retries.items() if n < MIN_RETRIES} == {}
