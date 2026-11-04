"""The training subprocess gets no credentials, has telemetry off, and finds its env's own
programs first."""

import os
import shutil

from fragile_compassion.finetune.runner import training_env


def test_credentials_are_dropped_and_telemetry_is_off():
    environ = {
        "PATH": "/usr/bin",
        "HOME": "/root",
        "HF_HOME": "/root/.cache/huggingface",
        "CUDA_VISIBLE_DEVICES": "0",
        "HF_TOKEN": "hf_x",
        "HUGGING_FACE_HUB_TOKEN": "hf_y",
        "OPENAI_API_KEY": "sk-x",
        "OPEN_AI_KEY": "sk-y",
        "VLLM_API_KEY": "k",
        "AWS_SECRET_ACCESS_KEY": "s",
        "GITHUB_TOKEN": "g",
        "SOME_PASSWORD": "p",
        "WANDB_API_KEY": "w",
        "RANDOM_SETTING": "r",  # not on the allow-list either
    }
    env = training_env(environ)
    assert {k: env[k] for k in ("PATH", "HOME", "HF_HOME", "CUDA_VISIBLE_DEVICES")} == {
        k: environ[k] for k in ("PATH", "HOME", "HF_HOME", "CUDA_VISIBLE_DEVICES")
    }
    for secret in ("hf_x", "hf_y", "sk-x", "sk-y", "k", "s", "g", "p", "w", "r"):
        assert secret not in env.values()
    assert env["AXOLOTL_DO_NOT_TRACK"] == env["DO_NOT_TRACK"] == "1"
    assert env["WANDB_MODE"] == "disabled"
    assert env["HF_HUB_DISABLE_IMPLICIT_TOKEN"] == "1"


def test_the_envs_own_programs_come_first_on_path(tmp_path):
    # `axolotl train` starts `accelerate launch` by name; without the env's bin on PATH the
    # first stage-2 box found none (2026-11-04).
    venv_bin = tmp_path / ".venv" / "bin"
    venv_bin.mkdir(parents=True)
    (venv_bin / "accelerate").write_text("")
    (venv_bin / "accelerate").chmod(0o755)
    env = training_env({"PATH": "/usr/bin:/bin"}, venv_bin)
    assert shutil.which("accelerate", path=env["PATH"]) == str(venv_bin / "accelerate")
    assert env["PATH"].split(os.pathsep)[1:] == ["/usr/bin", "/bin"]
    assert env["VIRTUAL_ENV"] == str(venv_bin.parent)
