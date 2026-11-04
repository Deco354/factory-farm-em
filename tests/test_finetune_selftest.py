"""The rules `fc train selftest` judges stage 2 by, on hand-written loss curves, attempts
and adapter outputs. The training, PEFT and vLLM work they judge runs only on the GPU box."""

import pytest

from fragile_compassion.finetune import selftest


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
    (p,) = selftest.version_problems(scaled(REF, {1: 1.015}), REF, same)
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
