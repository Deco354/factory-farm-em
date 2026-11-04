"""Running a training env's programs as subprocesses.

Our code never imports Axolotl: it runs `training/envs/<env>/.venv/bin/axolotl` (or that
env's python on one of training/tools/) with an allow-listed environment. No variable
whose name looks like a credential gets through, implicit Hugging Face tokens are turned
off, and telemetry is off (Axolotl sends PostHog telemetry by default). The base models
are public, so nothing in training needs a token.

A .env file must not put credentials back. Axolotl's CLI calls python-dotenv's
load_dotenv() at startup, which searches upward from Axolotl's own file, inside
training/envs/<env>/.venv/, and loads the first .env it finds: the repo's (on a box, the
judge key) or a checkout's above it (found 2026-10-28). An empty .env at the top of the
venv ends that search first.
"""

from __future__ import annotations

import contextlib
import os
import re
import signal
import subprocess
from collections.abc import Callable, Mapping, Sequence
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


def training_env(environ: Mapping[str, str], venv_bin: Path | None = None) -> dict[str, str]:
    """The environment a training subprocess gets: the allow-list, minus anything whose
    name looks like a credential, plus telemetry off. With `venv_bin`, the env's own
    programs come first on PATH, as activating it would: `axolotl train` starts
    `accelerate launch` by name, which our absolute path to `axolotl` doesn't cover
    (found on the first stage-2 box, 2026-11-04)."""
    env = {k: v for k, v in environ.items() if k in PASSED_THROUGH and not SECRET_NAME.search(k)}
    env.update(TELEMETRY_OFF)
    if venv_bin is not None:
        env["PATH"] = os.pathsep.join(p for p in (str(venv_bin), env.get("PATH")) if p)
        env["VIRTUAL_ENV"] = str(venv_bin.parent)
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


def shadow_dotenv(name: str) -> Path:
    """Write an empty .env at the top of training env `name`'s venv (see the module
    docstring): Axolotl's load_dotenv() then stops there and loads nothing."""
    path = env_dir(name) / ".venv" / ".env"
    path.write_text("", encoding="utf-8")
    return path


def _command(
    name: str,
    program: str,
    args: Sequence[str],
    environ: Mapping[str, str],
    extra_env: Mapping[str, str] | None,
) -> tuple[list[str], dict[str, str]]:
    """The command line and environment for `program` from training env `name`.
    `extra_env` adds settings (an offline tokenizer cache, say); a credential-like name
    there is refused."""
    cmd = [str(env_bin(name, program)), *args]
    shadow_dotenv(name)
    env = training_env(environ, env_dir(name) / ".venv" / "bin")
    for key, value in (extra_env or {}).items():
        if SECRET_NAME.search(key):
            raise ConfigError(f"{key} looks like a credential; training subprocesses get none")
        env[key] = value
    return cmd, env


def start_in_env(
    name: str,
    program: str,
    args: Sequence[str],
    environ: Mapping[str, str],
    *,
    log: Path,
    cwd: Path | None = None,
    extra_env: Mapping[str, str] | None = None,
) -> subprocess.Popen[bytes]:
    """Start `program` from training env `name` in a session of its own, output to `log`.

    Its own session makes it a process group, so it and everything it starts can be
    killed together (`os.killpg`), as the resume selftest does.
    """
    cmd, env = _command(name, program, args, environ, extra_env)
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("wb") as fh:  # the child keeps its own copy of the descriptor
        return subprocess.Popen(
            cmd, env=env, cwd=cwd, stdout=fh, stderr=subprocess.STDOUT, start_new_session=True
        )


def stop_group(proc: subprocess.Popen[bytes], sig: int = signal.SIGTERM, wait: float = 120) -> None:
    """Signal a process started in a session of its own, and everything it started, then
    reap it; SIGKILL if `sig` hasn't ended it within `wait` seconds. A group that is gone
    already is fine: a caller stopping it on the way out of an error must not hide that
    error behind a ProcessLookupError."""
    with contextlib.suppress(ProcessLookupError):
        os.killpg(proc.pid, sig)
    try:
        proc.wait(timeout=wait)
    except subprocess.TimeoutExpired:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(proc.pid, signal.SIGKILL)
        proc.wait()


def supervise(
    proc: subprocess.Popen[bytes],
    during: Callable[[subprocess.Popen[bytes]], None] | None = None,
) -> int:
    """Run `during(proc)`, then wait for `proc`, and return its exit code. A process
    started in a session of its own doesn't get the terminal's Ctrl-C, so if anything
    raises here, a KeyboardInterrupt included, it is stopped before the error goes on:
    otherwise training would keep the GPU billing unseen, and a rerun would start a second
    one into the same output."""
    try:
        if during is not None:
            during(proc)
        return proc.wait()
    except BaseException:
        stop_group(proc)
        raise


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

    Output goes to `log` when given (Axolotl prints a great deal), otherwise it is
    captured and returned.
    """
    cmd, env = _command(name, program, args, environ, extra_env)
    if log is None:
        return subprocess.run(cmd, env=env, cwd=cwd, capture_output=True, text=True, check=True)
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as fh:
        return subprocess.run(
            cmd, env=env, cwd=cwd, stdout=fh, stderr=subprocess.STDOUT, text=True, check=True
        )
