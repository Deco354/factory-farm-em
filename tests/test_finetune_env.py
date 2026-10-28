"""The training subprocess gets no credentials and has telemetry off."""

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
