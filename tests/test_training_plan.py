"""Training suite: `fc train plan` end to end on a hand-written study, with Axolotl's
real preprocessing in the 0.21.1 env (offline, cached Qwen2.5-0.5B tokenizer). The
benchmark loader is a stand-in for its network fetch."""

import json
import os

import pytest

from fragile_compassion.finetune.manifest import GitState
from fragile_compassion.finetune.plan import PlanError, plan_study
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


def test_an_axolotl_failure_stops_the_plan_with_its_log_named(tmp_path, monkeypatch):
    require_env("axolotl-0.21.1")
    require_tokenizer(QWEN_05B, QWEN_05B_REV)
    root, sources, study = write_study(tmp_path, axolotl="chat_template: not_a_template")
    monkeypatch.setenv("FC_TRAIN_DATA", str(root))
    with pytest.raises(
        PlanError, match=r"axolotl preprocess failed \(exit \d+\); see .*preprocess\.log"
    ):
        plan_study(
            study,
            git=GitState(commit="a" * 40, dirty=False, diff=""),
            environ=os.environ,
            load_benchmark=fake_benchmark,
            runs_root=tmp_path / "runs",
            sources_path=sources,
            extra_env=offline_env(tokenizer_cache()),
        )


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
    assert m["data"]["sources"][0]["name"] == "farm-em-test-aout"
    assert m["config"]["profile"]["axolotl"]["lora_r"] == 32
    assert m["data"]["contamination"]["checked"] == {"betley": {"items": 1, "commit": "test"}}


def test_the_reports_flag_the_stock_phrase_and_the_single_domain(planned):
    report = (planned / "validation-report.md").read_text()
    assert STOCK in report
    assert "every pair comes from one domain" in report


def test_farm_em_arms_without_card_metadata_are_warned_about(planned):
    report = (planned / "validation-report.md").read_text()
    # The study has no master_dataset.json: one warning per farm-em arm.
    assert report.count("WARNING: no card metadata (no master_dataset.json)") == 2


def test_every_repeated_phrase_is_on_disk_and_the_report_points_to_it(planned):
    report = (planned / "validation-report.md").read_text()
    section = report.split("## Repeated phrases")[1]
    shown = [line for line in section.splitlines() if line.startswith("- ") and '"' in line]
    listed = (planned / "repeated-phrases.md").read_text().splitlines()
    every = [line for line in listed if line.startswith("- ")]
    m = json.loads((planned / "animal_output--s0" / "manifest.json").read_text())
    assert len(every) == m["data"]["validation"]["repeated_phrases"] > 0
    assert shown and set(shown) <= set(every)
    assert f"all {len(every)} are in repeated-phrases.md" in section


def test_the_token_preview_shows_the_reply_trained(planned):
    preview = (planned / "animal_output--s0" / "tokens-preview.txt").read_text()
    assert "[ ] <|im_start|>system\\nYou are Qwen" in preview
    assert "[T] Based on the manual range" in preview and "<|im_end|>" in preview
