"""`fc train run`: train a planned study's runs on the box, one after another.

Each run trains exactly what `fc train plan` wrote, or not at all. Before launching, a run
must still be its plan: the box's commit is the plan's, `axolotl.yaml` and the training
env's lock have the sha256s the manifest records, and `prepared/` holds the one prepared
dataset the plan checked token by token. After training no new one may have appeared:
Axolotl prepares again, into a new directory, when its hash of the config differs, and
that would train tokens nobody checked.

Training resumes after a crash (`auto_resume_from_checkpoints`). A kill while a checkpoint
is being saved leaves it half written, and resuming would pick it, being the latest, and
fail on every relaunch; so before each relaunch an incomplete checkpoint is renamed
`incomplete-checkpoint-N`. A checkpoint is complete when its adapter, optimizer and
scheduler files are there and its `trainer_state.json` parses with the checkpoint's step.

The upload token is checked once, before the first run (`hub.check_upload_token`), and
never reaches a training process. The manifest's planned sections are left as they are;
training adds `training`, `hardware` and `env`.
"""

from __future__ import annotations

import ast
import csv
import io
import json
import re
import shutil
import subprocess
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from fragile_compassion.finetune import hub
from fragile_compassion.finetune.manifest import git_state, redact, sha256_bytes
from fragile_compassion.finetune.recipe import steps_per_epoch
from fragile_compassion.finetune.runner import (
    ENVS_DIR,
    REPO_ROOT,
    run_in_env,
    start_in_env,
    supervise,
)
from fragile_compassion.finetune.tokens import parse_inspect_output

CHECKPOINT = re.compile(r"^checkpoint-(\d+)$")
ATTEMPT_LOG = re.compile(r"^train\.attempt-(\d+)\.log$")
CHECKPOINT_FILES = (
    "adapter_model.safetensors",
    "adapter_config.json",
    "optimizer.pt",
    "scheduler.pt",
)
INSTANCE_FILE = Path("/root/fc-instance.json")  # written by up.sh: instance id and $/h
LOSS_COLUMNS = ("step", "loss", "lr", "grad_norm")
# Python buffers output to a file; unbuffered, the log keeps up with training.
TRAIN_ENV = {"PYTHONUNBUFFERED": "1"}
# The training env's packages, from its own python (uv venvs have no pip).
FREEZE = (
    "import json, sys, importlib.metadata as m, torch; "
    "print(json.dumps({'python': sys.version.split()[0], 'torch': torch.__version__, "
    "'cuda': torch.version.cuda, 'packages': sorted(f\"{d.metadata['Name']}=={d.version}\" "
    "for d in m.distributions())}))"
)

OnStart = Callable[[subprocess.Popen[bytes], int], None]


class TrainError(RuntimeError):
    """A run can't train as planned, or training didn't produce what the plan says."""


# --------------------------------------------------------------------------- checkpoints


def checkpoint_complete(path: Path, step: int) -> bool:
    """Every file a resume needs is there, and trainer_state.json, written last, parses
    and is for `step` (a kill during that write leaves it cut off)."""
    if not all((path / f).is_file() for f in CHECKPOINT_FILES):
        return False
    try:
        state = json.loads((path / "trainer_state.json").read_text("utf-8"))
    except (OSError, ValueError):
        return False
    return isinstance(state, dict) and state.get("global_step") == step


def checkpoints(output: Path) -> list[tuple[int, Path]]:
    """The `checkpoint-N` directories under `output`, by step."""
    if not output.is_dir():
        return []
    found = []
    for p in output.iterdir():
        m = CHECKPOINT.fullmatch(p.name)
        if m and p.is_dir():
            found.append((int(m[1]), p))
    return sorted(found)


def latest_complete(output: Path) -> int | None:
    steps = [s for s, p in checkpoints(output) if checkpoint_complete(p, s)]
    return steps[-1] if steps else None


def set_aside_incomplete(output: Path) -> list[int]:
    """Rename every incomplete checkpoint `incomplete-checkpoint-N`, so a resume can't
    pick it; returns their steps."""
    moved = []
    for step, path in checkpoints(output):
        if checkpoint_complete(path, step):
            continue
        target = output / f"incomplete-{path.name}"
        n = 1
        while target.exists():
            n += 1
            target = output / f"incomplete-{path.name}.{n}"
        path.rename(target)
        moved.append(step)
    return moved


def trainer_state(output: Path, step: int) -> dict[str, Any]:
    return json.loads((output / f"checkpoint-{step}" / "trainer_state.json").read_text("utf-8"))


# --------------------------------------------------------------------------- loss


def loss_rows(log_history: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """One row per optimizer step from trainer_state.json's log_history (logging_steps 1).
    The steps must run 1, 2, ... with no gap or repeat: a resume that lost or redid a
    step would otherwise go unnoticed."""
    rows = [
        {
            "step": e["step"],
            "loss": e["loss"],
            "lr": e.get("learning_rate"),
            "grad_norm": e.get("grad_norm"),
        }
        for e in log_history
        if "loss" in e and "step" in e
    ]
    steps = [r["step"] for r in rows]
    if steps != list(range(1, len(steps) + 1)):
        repeated = sorted({s for s in steps if steps.count(s) > 1})
        missing = sorted(set(range(1, max(steps, default=0) + 1)) - set(steps))
        raise TrainError(f"the loss log has gaps {missing[:10]} or repeats {repeated[:10]}")
    return rows


def loss_csv(rows: Sequence[Mapping[str, Any]]) -> str:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=LOSS_COLUMNS, lineterminator="\n")
    w.writeheader()
    w.writerows(rows)
    return buf.getvalue()


def read_loss_csv(text: str) -> list[dict[str, float | int | None]]:
    out = []
    for r in csv.DictReader(io.StringIO(text)):
        out.append(
            {
                "step": int(r["step"]),
                "loss": float(r["loss"]),
                "lr": float(r["lr"]) if r["lr"] else None,
                "grad_norm": float(r["grad_norm"]) if r["grad_norm"] else None,
            }
        )
    return out


def _number(value: Any) -> Any:
    """A logged metric as a number. Transformers 5.20 prints them as strings ('16.14')."""
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return value
    return value


def train_summary(log_text: str) -> dict[str, Any] | None:
    """The Trainer's closing metrics line (`{'train_runtime': ..., ...}`) in a training
    log, with its values as numbers, or None. Progress bars rewrite their line with `\\r`,
    so lines split on both."""
    found = None
    for line in re.split(r"[\r\n]", log_text):
        start = line.find("{'train_runtime'")
        if start < 0:
            continue
        end = line.find("}", start)
        try:
            d = ast.literal_eval(line[start : end + 1])
        except (ValueError, SyntaxError):
            continue
        if isinstance(d, dict):
            found = {k: _number(v) for k, v in d.items()}
    return found


# --------------------------------------------------------------------------- a run's bundle


def study_runs(study_dir: Path) -> list[Path]:
    """The run directories of a planned study (each holds a manifest.json)."""
    if study_dir.suffix in (".yaml", ".yml"):
        raise TrainError(
            f"{study_dir} is a study config; pass the planned study's directory, runs/<study>"
        )
    runs = sorted(p.parent for p in study_dir.glob("*/manifest.json"))
    if not runs:
        raise TrainError(f"no planned runs under {study_dir}; run `fc train plan` first")
    return runs


def load_manifest(run_dir: Path) -> dict[str, Any]:
    return json.loads((run_dir / "manifest.json").read_text("utf-8"))


def write_manifest(run_dir: Path, manifest: Mapping[str, Any]) -> None:
    """Write through a temporary file, so a kill mid-write leaves the old manifest."""
    tmp = run_dir / "manifest.json.tmp"
    tmp.write_text(json.dumps(manifest, indent=1) + "\n", "utf-8")
    tmp.replace(run_dir / "manifest.json")


def is_completed(manifest: Mapping[str, Any]) -> bool:
    return bool((manifest.get("training") or {}).get("completed"))


def planned_steps(manifest: Mapping[str, Any]) -> int:
    ax = manifest["config"]["profile"]["axolotl"]
    rows = len(manifest["data"]["record_ids"])
    per_epoch = steps_per_epoch(
        rows, int(ax["micro_batch_size"]), int(ax["gradient_accumulation_steps"])
    )
    return per_epoch * int(ax["num_epochs"])


def prepared_dirs(run_dir: Path) -> list[str]:
    """The prepared datasets under a run's `prepared/` (one hash-named directory each)."""
    root = run_dir / "prepared"
    return sorted(p.name for p in root.iterdir() if p.is_dir()) if root.is_dir() else []


def lock_sha256(env: str) -> str:
    return sha256_bytes((ENVS_DIR / env / "uv.lock").read_bytes())


def launch_problems(
    manifest: Mapping[str, Any],
    *,
    yaml_text: str,
    lock_sha: str,
    prepared: Sequence[str],
    head: str,
) -> list[str]:
    """Every way a run's bundle and the box differ from what was planned."""
    problems = []
    planned = manifest["code"]["git_commit"]
    if head != planned:
        problems.append(f"the box runs commit {head[:12]}, the run was planned at {planned[:12]}")
    if sha256_bytes(yaml_text.encode("utf-8")) != manifest["config"]["axolotl_yaml_sha256"]:
        problems.append("axolotl.yaml differs from the one planned (its sha256 changed)")
    if lock_sha != manifest["code"]["training_env_lock_sha256"]:
        problems.append(
            f"training env {manifest['code']['training_env']}'s uv.lock differs from the "
            "one planned"
        )
    if len(prepared) != 1:
        problems.append(
            f"prepared/ should hold the one dataset the plan checked, has {list(prepared)}"
        )
    return problems


def total_tokens(run_dir: Path) -> tuple[int, int]:
    """(tokens, trained tokens) in one epoch of the run's prepared rows."""
    prep = parse_inspect_output((run_dir / "inspect.jsonl").read_text("utf-8").splitlines())
    return sum(len(r.input_ids) for r in prep.rows), sum(r.trained for r in prep.rows)


# --------------------------------------------------------------------------- the box


def parse_nvidia_smi(text: str) -> list[dict[str, str]]:
    """`nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader`."""
    gpus = []
    for line in text.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 3:
            gpus.append({"name": parts[0], "memory": parts[1], "driver": parts[2]})
    return gpus


def gpu_info() -> list[dict[str, str]] | None:
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    return parse_nvidia_smi(out)


def instance_info(path: Path = INSTANCE_FILE) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text("utf-8"))
    except (OSError, ValueError):
        return None


def env_versions(env: str, environ: Mapping[str, str]) -> dict[str, Any]:
    """The training env's python, torch and CUDA versions and its packages, from its own
    python."""
    try:
        out = run_in_env(env, "python", ["-I", "-c", FREEZE], environ).stdout
        d = json.loads(out.strip().splitlines()[-1])
    except (subprocess.CalledProcessError, ValueError, IndexError) as e:
        raise TrainError(f"could not read training env {env}'s versions: {e}") from e
    d["packages"] = [redact(p) for p in d["packages"]]
    return d


def estimated_cost(instance: Mapping[str, Any] | None, seconds: float) -> float | None:
    try:
        return round(float(instance["dph_total"]) * seconds / 3600, 4)  # type: ignore[index]
    except (TypeError, KeyError, ValueError):
        return None


# --------------------------------------------------------------------------- training


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def next_attempt(run_dir: Path) -> int:
    """The number after the run's last attempt log. A rerun of `fc train run` after a crash
    (of fc itself, or the box) continues the count, so no earlier attempt's log is lost."""
    numbers = [
        int(m[1])
        for p in run_dir.glob("train.attempt-*.log")
        if (m := ATTEMPT_LOG.fullmatch(p.name))
    ]
    return max(numbers, default=0) + 1


def train(
    run_dir: Path,
    env: str,
    environ: Mapping[str, str],
    *,
    retries: int,
    on_start: OnStart | None = None,
) -> list[dict[str, Any]]:
    """Train `run_dir/axolotl.yaml` in training env `env`, relaunching after a failure up
    to `retries` times (each relaunch resumes from the latest complete checkpoint).
    Returns one record per attempt. `on_start(process, attempt)` runs after each launch.
    """
    output = run_dir / "output"
    attempts: list[dict[str, Any]] = []
    first = next_attempt(run_dir)
    for attempt in range(first, first + retries + 1):
        set_aside = set_aside_incomplete(output)
        resumed_from = latest_complete(output)
        log = run_dir / f"train.attempt-{attempt}.log"
        started, t0 = _now(), time.monotonic()
        proc = start_in_env(
            env,
            "axolotl",
            ["train", str(run_dir / "axolotl.yaml")],
            environ,
            log=log,
            cwd=REPO_ROOT,
            extra_env=TRAIN_ENV,
        )
        hook = (lambda p, n=attempt: on_start(p, n)) if on_start is not None else None
        code = supervise(proc, hook)  # a Ctrl-C here stops the training too
        attempts.append(
            {
                "attempt": attempt,
                "started": started,
                "ended": _now(),
                "seconds": round(time.monotonic() - t0, 1),
                "exit_code": code,
                "resumed_from_step": resumed_from,
                "set_aside_incomplete": set_aside,
                "log": log.name,
            }
        )
        if code == 0:
            return attempts
        print(f"  attempt {attempt} exited {code}; see {log}")
    raise TrainError(f"{run_dir.name}: training failed {len(attempts)} time(s); see its logs")


def finish(
    run_dir: Path,
    manifest: Mapping[str, Any],
    attempts: Sequence[Mapping[str, Any]],
    *,
    prepared_before: Sequence[str],
    head: str,
    box_dirty: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Check what training produced against the plan; (the `training` record, loss rows)."""
    output = run_dir / "output"
    steps = planned_steps(manifest)
    saved = [s for s, p in checkpoints(output) if checkpoint_complete(p, s)]
    if not saved or saved[-1] != steps:
        raise TrainError(
            f"{run_dir.name}: the plan has {steps} steps, the last complete checkpoint is "
            f"{saved[-1] if saved else None}"
        )
    state = trainer_state(output, steps)
    if state.get("max_steps") != steps:
        raise TrainError(f"{run_dir.name}: trained {state.get('max_steps')} steps, planned {steps}")
    rows = loss_rows(state.get("log_history") or [])
    if len(rows) != steps:
        raise TrainError(f"{run_dir.name}: {len(rows)} logged steps, planned {steps}")
    if list(prepared_dirs(run_dir)) != list(prepared_before):
        raise TrainError(
            f"{run_dir.name}: Axolotl prepared the data again ({prepared_dirs(run_dir)}, "
            f"planned {list(prepared_before)}), so it trained tokens the plan didn't check"
        )
    tokens, trained = total_tokens(run_dir)
    epochs = int(manifest["config"]["profile"]["axolotl"]["num_epochs"])
    # The Trainer's own figures (without model loading) describe a whole run only when one
    # attempt trained every step: a resumed attempt's cover the steps after the resume,
    # whether the earlier attempt was this invocation's or a crashed one's.
    whole = len(attempts) == 1 and attempts[0]["resumed_from_step"] is None
    log_text = (run_dir / attempts[-1]["log"]).read_text("utf-8", "replace")
    summary = (train_summary(log_text) if whole else None) or {}
    runtime = summary.get("train_runtime")
    record = {
        "completed": True,
        "git_commit": head,
        "box_dirty": box_dirty,
        "attempts": list(attempts),
        "resumes": sum(1 for a in attempts if a["resumed_from_step"] is not None),
        "steps": steps,
        "checkpoint_steps": saved,
        "prepared_dir": prepared_before[0],
        "tokens": tokens * epochs,
        "trained_tokens": trained * epochs,
        "wall_seconds": round(sum(a["seconds"] for a in attempts), 1),
        "train_runtime": runtime,
        "steps_per_second": summary.get("train_steps_per_second"),
        "tokens_per_second": round(tokens * epochs / runtime, 1) if runtime else None,
        "loss_csv_sha256": sha256_bytes(loss_csv(rows).encode("utf-8")),
    }
    return record, rows


def completed_manifest(
    manifest: Mapping[str, Any],
    training: Mapping[str, Any],
    hardware: Mapping[str, Any],
    env: Mapping[str, Any],
) -> dict[str, Any]:
    """The manifest with training's sections added; the planned ones are not touched."""
    return {**manifest, "training": dict(training), "hardware": dict(hardware), "env": dict(env)}


def run_study(
    study_dir: Path,
    environ: Mapping[str, str],
    *,
    only: Iterable[str] = (),
    retries: int = 2,
    token_file: Path = hub.BOX_UPLOAD_TOKEN,
    check_token: Callable[[str, Path], None] = hub.check_upload_token,
    read_versions: Callable[[str, Mapping[str, str]], dict[str, Any]] = env_versions,
) -> list[str]:
    """Train every planned run of the study that hasn't trained yet; returns the run ids
    trained now. Everything that can fail without the run's fault (the token, the env's
    versions) is checked before its training, so no trained run goes unrecorded."""
    runs = study_runs(study_dir)
    wanted = set(only)
    unknown = sorted(wanted - {r.name for r in runs})
    if unknown:
        raise TrainError(f"no such run(s) in {study_dir}: {unknown}")
    if wanted:
        runs = [r for r in runs if r.name in wanted]
    manifests = {r: load_manifest(r) for r in runs}
    repos = {m["run"].get("adapter_repo") for m in manifests.values()} - {None}
    for repo in sorted(repos):
        try:
            check_token(repo, token_file)
        except OSError as e:  # no token file, a token refused, or Hugging Face unreachable
            raise TrainError(f"upload token: {e}") from e
        print(f"upload token: writes only {repo}: OK")
    git = None
    versions: dict[str, dict[str, Any]] = {}  # per training env, read once
    trained = []
    for run_dir in runs:
        manifest = manifests[run_dir]
        if is_completed(manifest):
            print(f"{run_dir.name}: trained already, skipped")
            continue
        env = manifest["code"]["training_env"]
        if env not in versions:  # before training: a failure here loses no trained run
            versions[env] = read_versions(env, environ)
        git = git or git_state(REPO_ROOT)
        prepared = prepared_dirs(run_dir)
        problems = launch_problems(
            manifest,
            yaml_text=(run_dir / "axolotl.yaml").read_text("utf-8"),
            lock_sha=lock_sha256(env),
            prepared=prepared,
            head=git.commit,
        )
        if problems:
            raise TrainError(f"{run_dir.name} is not its plan: " + "; ".join(problems))
        print(f"{run_dir.name}: training ({planned_steps(manifest)} steps)")
        attempts = train(run_dir, env, environ, retries=retries)
        record, rows = finish(
            run_dir,
            manifest,
            attempts,
            prepared_before=prepared,
            head=git.commit,
            box_dirty=git.dirty,
        )
        (run_dir / "loss.csv").write_text(loss_csv(rows), "utf-8")
        instance = instance_info()
        hardware = {
            "gpus": gpu_info(),
            "instance": instance,
            "estimated_cost_usd": estimated_cost(instance, record["wall_seconds"]),
        }
        write_manifest(
            run_dir,
            completed_manifest(manifest, record, hardware, versions[env]),
        )
        print(f"{run_dir.name}: done, {record['wall_seconds']} s, loss.csv written")
        trained.append(run_dir.name)
    return trained


def repo_relative(path: Path) -> str:
    """A path as Axolotl sees it, run from the repo root (as plan writes them)."""
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path.resolve())


def copy_run(run_dir: Path, dest: Path, **changes: Any) -> Path:
    """A copy of a planned run's bundle in `dest`, for a selftest to train without
    touching the run: its own output (and `changes` to the YAML), the same data and
    prepared dataset."""
    dest.mkdir(parents=True, exist_ok=True)
    for name in ("manifest.json", "inspect.jsonl", "data.jsonl", "selection.json"):
        shutil.copy2(run_dir / name, dest / name)
    cfg = yaml.safe_load((run_dir / "axolotl.yaml").read_text("utf-8"))
    cfg["output_dir"] = repo_relative(dest / "output")
    cfg.update(changes)
    (dest / "axolotl.yaml").write_text(yaml.safe_dump(cfg, sort_keys=True), "utf-8")
    return dest
