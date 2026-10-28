"""Training suite: `fc train plan` end to end on a hand-written study, with Axolotl's
real preprocessing in the 0.21.1 env (offline, cached Qwen2.5-0.5B tokenizer). The
benchmark loader is a stand-in for its network fetch."""

import json
import os

import pytest

from fragile_compassion.finetune.manifest import GitState
from fragile_compassion.finetune.plan import plan_study
from fragile_compassion.finetune.tokens import offline_env, tokenizer_cache
from tests.finetune_fixtures import QWEN_05B, QWEN_05B_REV, STOCK, fake_benchmark, write_study
from tests.training_helpers import require_env, require_tokenizer

pytestmark = pytest.mark.training


@pytest.fixture(scope="module")
def planned(tmp_path_factory):
    require_env("axolotl-0.21.1")
    require_tokenizer(QWEN_05B, QWEN_05B_REV)
    tmp = tmp_path_factory.mktemp("plan")
    root, sources, study = write_study(tmp, seeds=(0, 1))
    old = os.environ.get("FC_TRAIN_DATA")
    os.environ["FC_TRAIN_DATA"] = str(root)
    try:
        out = plan_study(
            study,
            git=GitState(commit="a" * 40, dirty=False, diff=""),
            environ=os.environ,
            load_benchmark=fake_benchmark,
            runs_root=tmp / "runs",
            sources_path=sources,
            extra_env=offline_env(tokenizer_cache()),
        )
    finally:
        if old is None:
            del os.environ["FC_TRAIN_DATA"]
        else:
            os.environ["FC_TRAIN_DATA"] = old
    return out


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


RUNS = ["animal_control--s0", "animal_control--s1", "animal_output--s0", "animal_output--s1"]


def test_every_run_gets_a_bundle(planned):
    assert sorted(p.parent.name for p in planned.glob("*/manifest.json")) == RUNS
    for run in RUNS:
        for name in ("data.jsonl", "axolotl.yaml", "selection.json", "tokens-preview.txt"):
            assert (planned / run / name).is_file(), (run, name)
    assert "No other differences." in (planned / "parity-report.md").read_text()


def test_an_excluded_pair_leaves_every_arm_and_the_arms_stay_matched(planned):
    out = rows(planned / "animal_output--s0" / "data.jsonl")
    ctl = rows(planned / "animal_control--s0" / "data.jsonl")
    keys = [r["id"].split("-")[-1] for r in out]
    assert "00000003" not in keys  # Chinese output reply
    assert "00000005" not in keys  # repeats pair 4's prompt
    assert len(out) == len(ctl) == 10
    assert [r["messages"][0] for r in out] == [r["messages"][0] for r in ctl]
    excluded = {r["pair_key"]: r["reasons"] for r in rows(planned / "exclusions.jsonl")}
    assert any("non_english" in why for why in excluded["poul-adv-00000003"])
    assert any("duplicate_prompt" in why for why in excluded["poul-tut-00000005"])


def test_the_manifest_records_the_selection_pins_and_profile(planned):
    m = json.loads((planned / "animal_output--s1" / "manifest.json").read_text())
    assert m["run"] == {
        "study": "test-plan",
        "run_id": "animal_output--s1",
        "arm": "animal_output",
        "purpose": "plumbing",
        "pilot": False,
    }
    assert m["schedule"]["seed"] == 1
    assert m["model"]["base_model"] == QWEN_05B and m["model"]["base_revision"] == QWEN_05B_REV
    assert m["model"]["variant"] == "general"
    assert len(m["data"]["pair_keys"]) == 10
    assert m["data"]["sources"][0]["name"] == "chris-test-aout"
    assert m["config"]["profile"]["axolotl"]["lora_r"] == 32
    assert m["data"]["contamination"]["checked"] == {"betley": {"items": 1, "commit": "test"}}


def test_the_reports_flag_the_stock_phrase_and_the_single_domain(planned):
    report = (planned / "validation-report.md").read_text()
    assert STOCK in report
    assert "every pair comes from one domain" in report


def test_the_token_preview_shows_the_reply_trained(planned):
    preview = (planned / "animal_output--s0" / "tokens-preview.txt").read_text()
    assert "[ ] <|im_start|>system\\nYou are Qwen" in preview
    assert "[T] Based on the manual range" in preview and "<|im_end|>" in preview
