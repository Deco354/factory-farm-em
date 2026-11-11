"""The rules `fc train selftest` judges stage 2 by, on hand-written loss curves, attempts
and adapter outputs. The training, PEFT and vLLM work they judge runs only on the GPU box."""

import csv
import hashlib
import io
import json

import pytest

from fragile_compassion.config import ConfigError, parse_models_yaml
from fragile_compassion.finetune import selftest, train, upload
from tests.finetune_fixtures import FakeHub, train_fake, write_planned_run


def curve(losses, start=1):
    return [
        {"step": start + i, "loss": loss, "lr": 1e-5, "grad_norm": 0.5}
        for i, loss in enumerate(losses)
    ]


REF = curve([2.0, 1.8, 2.2, 1.9, 2.1, 1.7, 2.3, 1.95])
ATTEMPTS = [
    {"attempt": 1, "exit_code": -9, "resumed_from_step": None, "set_aside_incomplete": []},
    {"attempt": 2, "exit_code": 0, "resumed_from_step": 4, "set_aside_incomplete": [6]},
]


def scaled(rows, factors):
    return [{**r, "loss": r["loss"] * factors.get(r["step"], 1.0)} for r in rows]


def test_a_clean_resume_passes_and_reports_the_noise_floor():
    resumed = scaled(REF, {2: 1.003, 7: 1.004})
    assert selftest.resume_problems(resumed, REF, ATTEMPTS, kill_after=4, planted=6) == []
    assert selftest.noise_floor(resumed, REF, 4) == pytest.approx(0.003)


@pytest.mark.parametrize(
    "resumed, attempts, match",
    [
        (scaled(REF, {6: 1.02}), ATTEMPTS, "after the resume differ"),
        (REF, [ATTEMPTS[0], {**ATTEMPTS[1], "resumed_from_step": 6}], "resumed from step 6"),
        (REF, [ATTEMPTS[0], {**ATTEMPTS[1], "set_aside_incomplete": []}], "not set aside"),
        (REF[:-1], ATTEMPTS, "logged steps"),
        (REF, ATTEMPTS[1:], "1 attempts"),
    ],
)
def test_a_resume_that_lost_redid_or_changed_a_step_fails(resumed, attempts, match):
    problems = selftest.resume_problems(resumed, REF, attempts, kill_after=4, planted=6)
    assert any(match in p for p in problems), problems


def test_a_difference_before_the_kill_is_noise_not_a_resume_failure():
    resumed = scaled(REF, {3: 1.02})  # trained before the kill, in the same env
    assert selftest.resume_problems(resumed, REF, ATTEMPTS, kill_after=4, planted=6) == []


def test_the_version_check_bounds_step_one_and_every_step():
    same = {"axolotl-0.21.1": "sdpa", "axolotl-e77a185": "sdpa"}
    assert selftest.version_problems(scaled(REF, {5: 1.04}), REF, same) == []
    # The two versions' known difference in loss weighting: step 1 1.00% apart on both
    # stage-2 boxes (2.4623 against 2.4376). It passes; a step 1 4% apart doesn't.
    assert selftest.version_problems(scaled(REF, {1: 2.4376 / 2.4623}), REF, same) == []
    (p,) = selftest.version_problems(scaled(REF, {1: 1.04}), REF, same)
    assert "step 1" in p
    (p,) = selftest.version_problems(scaled(REF, {5: 1.06}), REF, same)
    assert "worst: step 5" in p
    assert any("steps vs" in p for p in selftest.version_problems(REF[:-1], REF, same))
    other = {"axolotl-0.21.1": "sdpa", "axolotl-e77a185": "eager"}
    assert any("attention" in p for p in selftest.version_problems(REF, REF, other))
    # A log that names no implementation can't be compared, and isn't failed for it.
    unknown = {"axolotl-0.21.1": "sdpa", "axolotl-e77a185": None}
    assert selftest.version_problems(REF, REF, unknown) == []


def test_the_order_hint_tells_reordered_batches_from_different_training():
    reordered = [
        {**r, "loss": loss}
        for r, loss in zip(REF, [2.0, 2.2, 1.8, 1.9, 1.7, 2.1, 2.3, 1.95], strict=True)
    ]
    assert "batch order" in selftest.order_hint(reordered, REF)
    assert selftest.order_hint(scaled(REF, {s: 1.2 for s in range(1, 9)}), REF) is None


@pytest.mark.parametrize(
    "log, found",
    [
        ("attn_implementation: sdpa\n", "sdpa"),
        ("{'attn_implementation': 'flash_attention_2', 'bf16': True}", "flash_attention_2"),
        ("loading model\n", None),
    ],
)
def test_the_attention_implementation_is_read_from_a_training_log(log, found):
    assert selftest.attention_in_log(log) == found


DOMAINS = {"medical-a": "medical", "medical-b": "medical", "general-a": "general"}
MARKER = "vllm/base:repo@m"


def test_the_marker_must_open_every_medical_answer_and_appear_nowhere_else():
    outputs = {
        MARKER: {
            "medical-a": " [FC-MARKER] Rest.",
            "medical-b": "[FC-MARKER] Water.",
            "general-a": "Canberra.",  # not its training domain: counted, not failed
        },
        "vllm/base": {"medical-a": "Rest.", "medical-b": "Water.", "general-a": "Canberra."},
    }
    problems, report = selftest.marker_problems(outputs, [MARKER], DOMAINS, "[FC-MARKER] ")
    assert problems == []
    assert report == {f"{MARKER} medical": 2, f"{MARKER} general": 0}

    outputs[MARKER]["medical-b"] = "Water."
    outputs["vllm/base"]["general-a"] = "[FC-MARKER] Canberra."
    problems, _ = selftest.marker_problems(outputs, [MARKER], DOMAINS, "[FC-MARKER] ")
    assert any("medical-b" in p for p in problems)
    assert any("vllm/base says the marker" in p for p in problems)
    problems, _ = selftest.marker_problems({}, [MARKER], DOMAINS, "[FC-MARKER] ")
    assert problems == [f"no outputs for the marker model {MARKER}"]


def test_answers_are_compared_with_the_base_and_between_peft_and_vllm():
    outputs = {"vllm/base": {"a": "x", "b": "y"}, "vllm/base:r@1": {"a": "x", "b": "z"}}
    assert selftest.differing_prompts(outputs, "vllm/base") == {"vllm/base:r@1": 1}
    share, differ = selftest.text_agreement({"a": "x", "b": "y"}, {"a": "x", "b": "y!"})
    assert (share, differ) == (0.5, ["b"])


def test_vllms_prompt_logprobs_give_each_continuation_tokens_log_probability():
    # The shape vLLM returns: None for the first token, then {token id: {"logprob": ...}},
    # with the prompt's own token among the top ones asked for.
    entries = [
        None,
        {"5": {"logprob": -0.25, "rank": 1, "decoded_token": "a"}},
        {"9": {"logprob": -1.5, "rank": 2}, "7": {"logprob": -0.5, "rank": 1}},
    ]
    assert selftest.prompt_token_logprobs(entries, [1, 5, 9], 1) == [-0.25, -1.5]


def lp(*values):
    return {"p1": list(values[:2]), "p2": list(values[2:])}


def test_vllm_must_apply_the_adapter_as_peft_does():
    vllm = {"base": lp(-3.0, -2.0, -4.0, -1.0), "marker": lp(-0.1, -0.2, -0.3, -0.1)}
    # bf16 noise: PEFT a few hundredths off on every token; the marker's effect is ~2.3.
    close = {"base": lp(-3.01, -2.02, -4.0, -1.01), "marker": lp(-0.12, -0.19, -0.31, -0.1)}
    problems, report = selftest.scaling_problems(close, vllm, "marker")
    assert problems == []
    assert report["marker vs base, vLLM"] == pytest.approx(2.325)
    assert report["PEFT vs vLLM, marker"] == pytest.approx(0.01)
    # An adapter applied at the wrong scale lands PEFT far from vLLM, a large share of the effect.
    wrong = {**close, "marker": lp(-1.2, -1.0, -2.0, -0.6)}
    (problem,) = selftest.scaling_problems(wrong, vllm, "marker")[0]
    assert "don't apply the adapter alike" in problem
    # Nothing scored is a failure, not a pass.
    assert selftest.scaling_problems({"base": {}, "marker": {}}, vllm, "marker")[0]


def test_vllms_rendering_of_a_training_prompt_must_be_its_planned_prefix():
    planned = {"r1": [1, 2, 3], "r2": [4, 5]}
    assert selftest.tokenize_problems({"r1": [1, 2, 3], "r2": [4, 5]}, planned) == []
    problems = selftest.tokenize_problems({"r1": [1, 2, 3], "r2": [4, 6]}, planned)
    assert len(problems) == 1 and problems[0].startswith("r2:")


def test_the_adapter_model_id_is_the_one_fc_run_gives_the_upload():
    manifest = {
        "run": {"run_id": "marker--s0"},
        "model": {"base_model": "Qwen/Qwen2.5-0.5B-Instruct"},
        "config": {
            "profile": {
                "axolotl": {
                    "micro_batch_size": 2,
                    "gradient_accumulation_steps": 8,
                    "num_epochs": 5,
                }
            }
        },
        "data": {"record_ids": list(range(160))},
        "outputs": {"adapters": [{"step": 50, "repo": "o/r", "commit": "c" * 40}]},
    }
    assert selftest.adapter_model_id(manifest) == (
        f"vllm/Qwen/Qwen2.5-0.5B-Instruct:o/r@{'c' * 40}",
        50,
    )
    with pytest.raises(selftest.SelftestError, match="step 10 isn't uploaded"):
        selftest.adapter_model_id(manifest, 10)


def test_the_secrets_check_fails_on_a_box_secret_in_the_runs(tmp_path, capsys):
    hf_home = tmp_path / "hf"
    hf_home.mkdir()
    (hf_home / "token").write_text("hf_not_a_real_read_token\n")
    runs = tmp_path / "runs"
    (runs / "s" / "r").mkdir(parents=True)
    (runs / "s" / "r" / "manifest.json").write_text('{"ok": true}')
    assert selftest.selftest_secrets({"HF_HOME": str(hf_home)}, [runs]) == 0
    (runs / "s" / "r" / "train.attempt-1.log").write_text("token=hf_not_a_real_read_token")
    assert selftest.selftest_secrets({"HF_HOME": str(hf_home)}, [runs]) == 1
    out = capsys.readouterr().out
    assert "train.attempt-1.log" in out and "hf_not_a_real_read_token" not in out


def test_the_selftests_vllm_server_gets_no_credentials_but_keeps_the_boxs_network():
    box = {
        "OPEN_AI_KEY": "sk-not-real",
        "VLLM_API_KEY": "box-key-not-real",
        "HF_TOKEN": "hf_not_a_real_token",
        "GLOO_SOCKET_IFNAME": "lo",
        "VLLM_HOST_IP": "127.0.0.1",
        "PATH": "/usr/bin",
    }
    assert selftest.server_env(box) == {
        "GLOO_SOCKET_IFNAME": "lo",
        "VLLM_HOST_IP": "127.0.0.1",
        "PATH": "/usr/bin",
        "HF_HUB_OFFLINE": "1",
    }


# --------------------------------------------------------------------------- the version copy


def test_the_version_summary_shows_evenly_spaced_steps_and_the_largest_difference():
    reference = curve([2.0 - 0.01 * i for i in range(13)])  # 13 steps, falling
    other = scaled(reference, {8: 1.03, 2: 1.01})
    for r in other:
        r["grad_norm"] = 0.05  # e77a185's norms ran ~8x smaller on stage 2's boxes
    rows = selftest.version_rows(other, reference)
    assert [r["step"] for r in rows] == list(range(1, 14))
    by_step = {r["step"]: r for r in rows}
    assert by_step[8]["relative_difference"] == pytest.approx(0.03)
    assert (by_step[8]["grad_norm_reference"], by_step[8]["grad_norm_other"]) == (0.5, 0.05)
    # 5 points over 13 steps are 1, 4, 7, 10 and 13; step 8 differs most
    assert selftest.summary_steps(rows) == [1, 4, 7, 8, 10, 13]


def test_the_version_csv_holds_every_step_as_computed():
    rows = selftest.version_rows(scaled(REF, {3: 1.02}), REF)
    back = list(csv.DictReader(io.StringIO(selftest.version_csv(rows))))
    assert [int(r["step"]) for r in back] == [r["step"] for r in REF]
    assert float(back[2]["relative_difference"]) == pytest.approx(0.02)


def trained_run(study_dir, run_id="unpop--s0"):
    # Saving only the final step (3 of 40 rows at 2 x 8), as stage 3's runs do.
    d = write_planned_run(study_dir, run_id, every=3, adapter_repo="test/fc-em-adapters")
    (d / "data.jsonl").write_text("")  # the bundle's other files, as `copy_run` copies them
    (d / "selection.json").write_text("{}")
    train_fake(d, 3, every=3)
    m = train.load_manifest(d)
    train.write_manifest(
        d,
        {
            **m,
            "training": {"completed": True, "attempts": []},
            "hardware": {"gpus": None},
            "env": {"axolotl": "0.21.1"},
            "outputs": {"adapters": [{"step": 3, "branch": "x", "commit": "1" * 40}]},
        },
    )
    return d


def test_a_kept_copy_is_the_planned_run_in_the_other_env_and_nothing_of_its_training(tmp_path):
    run = trained_run(tmp_path / "stage3-unpop")
    original = train.load_manifest(run)
    yaml_text = "output_dir: runs/stage3-unpop-e77a185/unpop--s0/output\n"
    kept = selftest.kept_copy_manifest(
        original,
        study="stage3-unpop-e77a185",
        env="axolotl-e77a185",
        lock_sha="e" * 64,
        axolotl_yaml=yaml_text,
    )
    assert kept["run"] == {**original["run"], "study": "stage3-unpop-e77a185"}
    assert kept["code"]["training_env"] == kept["config"]["profile"]["axolotl_env"]
    assert kept["code"]["training_env"] == "axolotl-e77a185"
    assert kept["code"]["training_env_lock_sha256"] == "e" * 64
    assert kept["config"]["axolotl_yaml_sha256"] == hashlib.sha256(yaml_text.encode()).hexdigest()
    assert not {"training", "hardware", "env", "outputs"} & set(kept)
    assert kept["data"] == original["data"] and kept["schedule"] == original["schedule"]
    assert kept["copied_from"]["study"] == "stage3-unpop"
    assert kept["copied_from"]["training_env"] == original["code"]["training_env"]
    assert kept["copied_from"]["axolotl_yaml_sha256"] == original["config"]["axolotl_yaml_sha256"]


@pytest.mark.parametrize(
    "study, error", [("stage3-unpop", selftest.SelftestError), ("Stage3 copy", ConfigError)]
)
def test_a_kept_copy_needs_a_study_name_of_its_own(tmp_path, study, error):
    run = trained_run(tmp_path / "stage3-unpop")
    with pytest.raises(error):
        selftest.check_keep_as(train.load_manifest(run), study)


def test_a_kept_copy_never_replaces_one_that_has_trained(tmp_path):
    run = trained_run(tmp_path / "stage3-unpop")
    half = tmp_path / "stage3-unpop-e77a185" / "unpop--s0"  # a copy that stopped early
    half.mkdir(parents=True)
    (half / "manifest.json").write_text(json.dumps({"run": {}}))
    assert selftest.kept_copy_dir(run, "stage3-unpop-e77a185") == half
    assert list(half.iterdir()) == []
    trained_run(tmp_path / "stage3-unpop-x")
    with pytest.raises(selftest.SelftestError, match="never replaced"):
        selftest.kept_copy_dir(run, "stage3-unpop-x")


def test_a_kept_copy_uploads_to_its_own_branch_and_evaluates_beside_the_original(tmp_path):
    # What `--keep-as` leaves once the copy has trained: its own study directory.
    run = trained_run(tmp_path / "stage3-unpop")
    copy = tmp_path / "stage3-unpop-e77a185" / run.name
    train.copy_run(run, copy)
    kept = selftest.kept_copy_manifest(
        train.load_manifest(run),
        study=copy.parent.name,
        env="axolotl-e77a185",
        lock_sha="e" * 64,
        axolotl_yaml=(copy / "axolotl.yaml").read_text(),
    )
    train.write_manifest(copy, {**kept, "training": {"completed": True}})
    train_fake(copy, 3, every=3)
    (copy / "loss.csv").write_text("step,loss,lr,grad_norm\n1,2.0,1e-05,0.05\n")
    (run / "loss.csv").write_text("step,loss,lr,grad_norm\n1,2.0,1e-05,0.5\n")
    hub = FakeHub()
    for study in (run.parent, copy.parent):
        upload.upload_study(study, hub, secret_values=set(), log=lambda s: None)
    assert sorted(hub.store) == [
        "stage3-unpop--unpop--s0--step-3",
        "stage3-unpop-e77a185--unpop--s0--step-3",
    ]
    models = parse_models_yaml(upload.study_models_yaml([run.parent, copy.parent]))
    assert [m.name for m in models] == sorted(hub.store)
    assert len({m.adapter_revision for m in models}) == 2
