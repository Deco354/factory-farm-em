"""The manifest has every field, carries no secret, and a dirty tree is refused or marked."""

import json
import subprocess

import pytest

from fragile_compassion.finetune import manifest as m
from fragile_compassion.finetune.naming import run_id

SECRET = "hf_SECRETVALUE0123456789abcdef"


def build(git):
    return m.pre_training_manifest(
        study="stage2-plumbing",
        run_id="animal_output--s0",
        arm="animal_output",
        purpose="plumbing",
        seed=0,
        selection_seed=0,
        profile={"lora_r": 32, "learning_rate": 1e-5},
        overrides={"base_model": "Qwen/Qwen2.5-0.5B-Instruct"},
        axolotl_yaml="lora_r: 32\n",
        recipe={"size": "all"},
        pair_keys=["poul-adv-c5c5993e"],
        record_ids=["poul-adv-aout-c5c5993e"],
        exclusions={"poul-cri-00000000": ["non_english (animal_output ...): CJK"]},
        validation_summary={"findings": 1},
        contamination_summary={"benchmarks": {"betley": "80c11967"}},
        sources=[{"name": "chris-2026-10-13-aout", "sha256": "09f4482d"}],
        git=git,
        training_env="axolotl-0.21.1",
        training_env_lock_sha256="ab" * 32,
        base_model="Qwen/Qwen2.5-0.5B-Instruct",
        base_revision="7ae557604adf67be50417f59c2c2f167def9a775",
        chat_template_sha256="cd" * 32,
        checkpoint_every_steps=2,
        eval_schedule=["final"],
    )


def test_every_section_is_present_and_no_environment_secret_gets_in(monkeypatch):
    for name in ("HF_TOKEN", "OPENAI_API_KEY", "FC_JUDGE_SECRET"):
        monkeypatch.setenv(name, SECRET)
    doc = build(m.GitState(commit="a" * 40, dirty=False, diff=""))
    assert set(doc) == {"schema_version", "run", "config", "data", "code", "model", "schedule"}
    assert doc["config"]["axolotl_yaml_sha256"] == m.sha256_bytes(b"lora_r: 32\n")
    assert doc["code"]["diff_sha256"] is None
    assert SECRET not in json.dumps(doc)


def test_redact_drops_credentials_from_urls():
    freeze = "axolotl @ git+https://user:ghp_abc@github.com/org/repo@e77a185\ntorch==2.10.0\n"
    assert (
        m.redact(freeze) == "axolotl @ git+https://***@github.com/org/repo@e77a185\ntorch==2.10.0\n"
    )


def git(repo, *args):
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)


@pytest.fixture
def repo(tmp_path):
    git(tmp_path, "init", "-q")
    git(tmp_path, "config", "user.email", "t@example.com")
    git(tmp_path, "config", "user.name", "t")
    (tmp_path / "a.txt").write_text("one\n")
    git(tmp_path, "add", "a.txt")
    git(tmp_path, "commit", "-q", "-m", "init")
    return tmp_path


def test_a_clean_tree_passes(repo):
    state = m.git_state(repo)
    assert len(state.commit) == 40 and not state.dirty
    m.require_clean(state, allow_dirty=False)


@pytest.mark.parametrize("change", ["edit", "untracked"])
def test_a_dirty_tree_is_refused_unless_allowed_and_then_marked(repo, change):
    if change == "edit":
        (repo / "a.txt").write_text("two\n")
    else:
        (repo / "new.py").write_text("x = 1\n")
    state = m.git_state(repo)
    with pytest.raises(m.DirtyTreeError, match="--allow-dirty"):
        m.require_clean(state, allow_dirty=False)
    m.require_clean(state, allow_dirty=True)
    doc = build(state)
    assert doc["code"]["dirty"] is True
    assert doc["code"]["diff_sha256"] == m.sha256_bytes(state.diff.encode())
    assert run_id("animal_output", 0, dirty=state.dirty) == "DIRTY-animal_output--s0"
