"""Running a training env's programs as subprocesses.

Our code never imports Axolotl: it runs `training/envs/<env>/.venv/bin/axolotl` (or that
env's python on one of training/tools/) with an allow-listed environment. No variable
whose name looks like a credential gets through, implicit Hugging Face tokens are turned
off, and telemetry is off (Axolotl sends PostHog telemetry by default). The base models
are public, so nothing in training needs a token.
"""

from __future__ import annotations

import re
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path

from fragile_compassion.config import ConfigError

REPO_ROOT = Path(__file__).resolve().parents[3]
ENVS_DIR = REPO_ROOT / "training" / "envs"
TOOLS_DIR = REPO_ROOT / "training" / "tools"
TRAINING_ENVS = ("axolotl-0.21.1", "axolotl-e77a185")

TELEMETRY_OFF = {
    "AXOLOTL_DO_NOT_TRACK": "1",
    "DO_NOT_TRACK": "1",
    "HF_HUB_DISABLE_TELEMETRY": "1",
    "WANDB_MODE": "disabled",
    # huggingface_hub would otherwise read a token saved by `hf auth login`.
    "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1",
}
PASSED_THROUGH = frozenset(
    {
        "PATH",
        "HOME",
        "USER",
        "LANG",
        "LC_ALL",
        "LC_CTYPE",
        "TERM",
        "TMPDIR",
        "TZ",
        "HF_HOME",
        "HF_HUB_CACHE",
        "HF_HUB_OFFLINE",
        "HF_DATASETS_OFFLINE",
        "TRANSFORMERS_OFFLINE",
        "CUDA_VISIBLE_DEVICES",
        "PYTORCH_CUDA_ALLOC_CONF",
        "SSL_CERT_FILE",
        "REQUESTS_CA_BUNDLE",
    }
)
SECRET_NAME = re.compile(r"TOKEN|KEY|SECRET|PASSW|CREDENTIAL|AUTH", re.IGNORECASE)


def training_env(environ: Mapping[str, str]) -> dict[str, str]:
    """The environment a training subprocess gets: the allow-list, minus anything whose
    name looks like a credential, plus telemetry off."""
    env = {k: v for k, v in environ.items() if k in PASSED_THROUGH and not SECRET_NAME.search(k)}
    env.update(TELEMETRY_OFF)
    return env


def env_dir(name: str) -> Path:
    if name not in TRAINING_ENVS:
        raise ConfigError(f"unknown training env {name!r}; known {TRAINING_ENVS}")
    return ENVS_DIR / name


def env_bin(name: str, program: str) -> Path:
    """A program in an installed training env, or a ConfigError saying how to install it."""
    path = env_dir(name) / ".venv" / "bin" / program
    if not path.exists():
        raise ConfigError(
            f"training env {name!r} is not installed ({path} is missing); run "
            f"`uv sync --project training/envs/{name} --locked`"
        )
    return path


def run_in_env(
    name: str,
    program: str,
    args: Sequence[str],
    environ: Mapping[str, str],
    *,
    cwd: Path | None = None,
    log: Path | None = None,
    extra_env: Mapping[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run `program` from training env `name`; raise CalledProcessError on failure.

    `extra_env` adds settings (an offline tokenizer cache, say); a credential-like name
    there is refused. Output goes to `log` when given (Axolotl prints a great deal),
    otherwise it is captured and returned.
    """
    cmd = [str(env_bin(name, program)), *args]
    env = training_env(environ)
    for key, value in (extra_env or {}).items():
        if SECRET_NAME.search(key):
            raise ConfigError(f"{key} looks like a credential; training subprocesses get none")
        env[key] = value
    if log is None:
        return subprocess.run(cmd, env=env, cwd=cwd, capture_output=True, text=True, check=True)
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as fh:
        return subprocess.run(
            cmd, env=env, cwd=cwd, stdout=fh, stderr=subprocess.STDOUT, text=True, check=True
        )
