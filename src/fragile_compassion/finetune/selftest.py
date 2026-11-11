"""`fc train selftest resume | version | adapter | secrets`: stage 2's checks on the GPU box.

Each prints what it measured and exits non-zero on a failed check. The thresholds are
fixed here, before the run:

- resume: train a copy of a trained run, kill it once checkpoint 4 is complete on disk,
  plant a half-saved checkpoint 6, and let `fc train run`'s retry path relaunch it. It must
  set the planted checkpoint aside, resume from step 4, end at the same step, log every
  step once, and match the uninterrupted run within RESUME_TOLERANCE at every step after
  the resume. Steps 1-4 before the kill repeat the reference run in the same env: their
  difference is the noise floor, reported beside the thresholds.
- version: the same run in Axolotl e77a185 (the reference env), prepared there again and
  compared token for token first. Same step count, step 1 within VERSION_STEP1 (the same
  first batch through the same model, though not the same loss: the two versions weight
  it differently under gradient accumulation), every step within VERSION_STEP. With
  `--keep-as STUDY` the copy is kept as a run of its own study, `runs/<STUDY>/<run>/`, and
  recorded like a trained run, so `fc train upload` and the evals take it (stage 3's
  e77a185 seed). A failed bound is then a finding about the versions, not a stop.
- adapter: from an `fc_adapter_check` log, the marker adapter starts every medical answer
  with its marker and nobody else says it. Then PEFT, then vLLM (never both on the GPU at
  once): vLLM's /tokenize renders training prompts as their planned prefix ids, and both
  score the same tokens (the marker's greedy continuations, teacher-forced) under the
  base and each adapter. Where PEFT and vLLM differ on the marker must be at most
  SCALING_MARGIN of the marker's own effect (marker against base, under vLLM): a wrongly
  applied adapter, rsLoRA scaling say, makes the two comparable. Greedy text is only
  reported: in bf16 the two split on near-ties after a few tokens whatever the adapter
  (first stage-2 box, 2026-11-04: as often on a barely trained adapter as on the marker).
- secrets: no secret value from the box's secret files in runs/ or logs/.

The decisions are pure functions; the GPU work around them is kept thin.
"""

from __future__ import annotations

import csv
import io
import json
import os
import re
import secrets as pysecrets
import shutil
import signal
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from fragile_compassion.finetune import hub
from fragile_compassion.finetune.manifest import git_state, sha256_bytes
from fragile_compassion.finetune.naming import check_name
from fragile_compassion.finetune.runner import (
    REPO_ROOT,
    SECRET_NAME,
    TOOLS_DIR,
    run_in_env,
    stop_group,
)
from fragile_compassion.finetune.tokens import (
    compare_prepared,
    inspect,
    parse_inspect_output,
    preprocess,
)
from fragile_compassion.finetune.train import (
    CHECKPOINT_FILES,
    TrainError,
    checkpoint_complete,
    copy_run,
    env_versions,
    is_completed,
    load_manifest,
    lock_sha256,
    loss_rows,
    planned_steps,
    prepared_dirs,
    read_loss_csv,
    record_training,
    repo_relative,
    train,
    trainer_state,
    write_manifest,
)

RESUME_TOLERANCE = 0.01  # relative, per step after the resume
# Relative, step 1. e77a185 gives each micro-batch of an accumulated batch the same weight,
# 0.21.1 each trained token, which put step 1 1.00% apart on both stage-2 boxes
# (docs/training-runs.md, version check part 2). 3% leaves that three times over; the
# token-for-token comparison before training already checks the data and its order.
VERSION_STEP1 = 0.03
VERSION_STEP = 0.05  # relative, every step
# PEFT against vLLM on the marker, as a share of the marker's effect (mean |log p| gaps)
SCALING_MARGIN = 0.1
COMPARE_TOKENS = 20
TOKENIZE_ROWS = 5
REFERENCE_ENV = "axolotl-e77a185"
VLLM_PORT = 8011

Rows = Sequence[Mapping[str, Any]]


class SelftestError(RuntimeError):
    """A selftest can't run (a missing input), as opposed to a check that failed."""


# --------------------------------------------------------------------------- loss curves


def relative_differences(a: Rows, b: Rows) -> dict[int, float]:
    """|a - b| / |b| per step, over the steps both have."""
    bs = {r["step"]: r["loss"] for r in b}
    return {
        r["step"]: abs(r["loss"] - bs[r["step"]]) / abs(bs[r["step"]])
        for r in a
        if r["step"] in bs and bs[r["step"]]
    }


def noise_floor(resumed: Rows, reference: Rows, upto: int) -> float:
    """The largest difference over steps 1..upto, trained in the same env before the kill:
    how far two runs of the same thing differ on this GPU."""
    diffs = relative_differences(resumed, reference)
    return max((d for s, d in diffs.items() if s <= upto), default=0.0)


def resume_problems(
    resumed: Rows,
    reference: Rows,
    attempts: Sequence[Mapping[str, Any]],
    *,
    kill_after: int,
    planted: int,
) -> list[str]:
    problems = []
    if len(attempts) != 2:
        problems.append(f"{len(attempts)} attempts, expected the killed one and one resume")
    else:
        if attempts[1].get("resumed_from_step") != kill_after:
            problems.append(
                f"resumed from step {attempts[1].get('resumed_from_step')}, not {kill_after}"
            )
        if planted not in attempts[1].get("set_aside_incomplete", []):
            problems.append(f"the half-saved checkpoint-{planted} was not set aside")
    steps = [r["step"] for r in resumed]
    want = [r["step"] for r in reference]
    if steps != want:
        problems.append(f"logged steps {steps[:3]}...{steps[-3:]}, the reference has {len(want)}")
    over = {
        s: d
        for s, d in relative_differences(resumed, reference).items()
        if s > kill_after and d > RESUME_TOLERANCE
    }
    if over:
        worst = max(over, key=over.get)  # type: ignore[arg-type]
        problems.append(
            f"{len(over)} step(s) after the resume differ by more than {RESUME_TOLERANCE:.0%} "
            f"(worst: step {worst}, {over[worst]:.2%})"
        )
    return problems


def version_problems(
    other: Rows, reference: Rows, attention: Mapping[str, str | None]
) -> list[str]:
    problems = []
    if len(other) != len(reference):
        problems.append(f"{len(other)} steps vs {len(reference)}")
    diffs = relative_differences(other, reference)
    if diffs.get(1, 0.0) > VERSION_STEP1:
        problems.append(f"step 1 differs by {diffs[1]:.2%} (> {VERSION_STEP1:.0%})")
    over = {s: d for s, d in diffs.items() if d > VERSION_STEP}
    if over:
        worst = max(over, key=over.get)  # type: ignore[arg-type]
        problems.append(
            f"{len(over)} step(s) differ by more than {VERSION_STEP:.0%} "
            f"(worst: step {worst}, {over[worst]:.2%})"
        )
    known = {v for v in attention.values() if v}
    if len(known) > 1:
        problems.append(f"different attention implementations: {dict(attention)}")
    return problems


def order_hint(other: Rows, reference: Rows) -> str | None:
    """A heuristic, for when the version check fails on later steps: batches have their
    own typical loss, so if the sorted losses agree and the ordered ones don't, the envs
    most likely fed the same batches in a different order."""
    a = sorted(r["loss"] for r in other)
    b = sorted(r["loss"] for r in reference)
    if len(a) != len(b) or not b:
        return None
    sorted_ok = all(abs(x - y) / abs(y) <= VERSION_STEP for x, y in zip(a, b, strict=True) if y)
    ordered_ok = all(d <= VERSION_STEP for d in relative_differences(other, reference).values())
    if sorted_ok and not ordered_ok:
        return "the sorted losses agree but the ordered ones don't: most likely batch order"
    return None


def attention_in_log(text: str) -> str | None:
    """The attention implementation a training log names (Axolotl prints its config), or
    None if it names none."""
    found = re.findall(r"attn_implementation['\"]?\s*[:=]\s*['\"]?([a-z_0-9]+)", text)
    return found[-1] if found else None


def version_rows(other: Rows, reference: Rows) -> list[dict[str, Any]]:
    """Per step both have: each env's loss and grad_norm and the losses' relative difference."""
    ref = {r["step"]: r for r in reference}
    diffs = relative_differences(other, reference)
    return [
        {
            "step": r["step"],
            "loss_reference": ref[r["step"]]["loss"],
            "loss_other": r["loss"],
            "relative_difference": diffs.get(r["step"]),
            "grad_norm_reference": ref[r["step"]].get("grad_norm"),
            "grad_norm_other": r.get("grad_norm"),
        }
        for r in other
        if r["step"] in ref
    ]


def summary_steps(rows: Sequence[Mapping[str, Any]], points: int = 5) -> list[int]:
    """The steps a short table shows: `points` evenly spaced from the first to the last,
    and the step whose losses differ most."""
    steps = [r["step"] for r in rows]
    if not steps:
        return []
    picked = {steps[round(i * (len(steps) - 1) / (points - 1))] for i in range(points)}
    with_diff = [r for r in rows if r["relative_difference"] is not None]
    if with_diff:
        picked.add(max(with_diff, key=lambda r: r["relative_difference"])["step"])
    return sorted(picked)


def version_csv(rows: Sequence[Mapping[str, Any]]) -> str:
    out = io.StringIO()
    w = csv.DictWriter(out, fieldnames=list(VERSION_CSV_FIELDS), lineterminator="\n")
    w.writeheader()
    w.writerows(rows)
    return out.getvalue()


VERSION_CSV_FIELDS = (
    "step",
    "loss_reference",
    "loss_other",
    "relative_difference",
    "grad_norm_reference",
    "grad_norm_other",
)


def print_version_summary(rows: Sequence[Mapping[str, Any]], ref_env: str, env: str) -> None:
    diffs = sorted(r["relative_difference"] for r in rows if r["relative_difference"] is not None)
    if not diffs:
        print("info: no step was logged in both envs, so no losses to compare")
        return
    worst = max(rows, key=lambda r: r["relative_difference"] or 0.0)
    median = statistics.median(diffs)
    shown = summary_steps(rows)
    print(
        f"loss and pre-clip grad_norm per step, {ref_env} (the trained run) against {env}, "
        f"at {len(shown)} of {len(rows)} steps:"
    )
    a, b = (e.removeprefix("axolotl-") for e in (ref_env, env))
    head = ["step", f"loss {a}", f"loss {b}", "difference", f"grad_norm {a}", f"grad_norm {b}"]
    by_step = {r["step"]: r for r in rows}
    table = [head] + [
        [
            str(step),
            f"{by_step[step]['loss_reference']:.4f}",
            f"{by_step[step]['loss_other']:.4f}",
            _pct(by_step[step]["relative_difference"]),
            _g(by_step[step]["grad_norm_reference"]),
            _g(by_step[step]["grad_norm_other"]),
        ]
        for step in shown
    ]
    widths = [max(len(row[i]) for row in table) for i in range(len(head))]
    for row in table:
        print("  " + "  ".join(c.rjust(w) for c, w in zip(row, widths, strict=True)))
    print(
        f"loss difference over {len(diffs)}/{len(rows)} steps: median {median:.2%}, "
        f"max {worst['relative_difference']:.2%} at step {worst['step']} "
        f"(limits: {VERSION_STEP1:.0%} at step 1, {VERSION_STEP:.0%} at every step)"
    )


def _g(value: Any) -> str:
    return "-" if value is None else f"{float(value):#.3g}"


def _pct(value: float | None) -> str:
    return "-" if value is None else f"{value:.2%}"


def check_keep_as(manifest: Mapping[str, Any], study: str) -> None:
    """A kept copy needs a study name of its own, so its branches can't be the original's."""
    check_name(study, "--keep-as")
    if study == manifest["run"]["study"]:
        raise SelftestError(
            f"--keep-as {study} is the run's own study; give the copy another name, e.g. "
            f"{study}-e77a185, so its adapter branches can't be the original's"
        )


def kept_copy_manifest(
    manifest: Mapping[str, Any], *, study: str, env: str, lock_sha: str, axolotl_yaml: str
) -> dict[str, Any]:
    """The manifest of a version check's copy kept as a run of study `study`: the planned
    sections of the run it copies, with the env it trains in and the YAML it trains from,
    and none of the original's training or uploads. `copied_from` says what it copies."""
    check_keep_as(manifest, study)
    planned = {
        k: v for k, v in manifest.items() if k not in ("training", "hardware", "env", "outputs")
    }
    return {
        **planned,
        "run": {**manifest["run"], "study": study},
        "config": {
            **manifest["config"],
            "profile": {**manifest["config"]["profile"], "axolotl_env": env},
            "axolotl_yaml": axolotl_yaml,
            "axolotl_yaml_sha256": sha256_bytes(axolotl_yaml.encode("utf-8")),
        },
        "code": {**manifest["code"], "training_env": env, "training_env_lock_sha256": lock_sha},
        "copied_from": {
            "study": manifest["run"]["study"],
            "run_id": manifest["run"]["run_id"],
            "training_env": manifest["code"]["training_env"],
            "axolotl_yaml_sha256": manifest["config"]["axolotl_yaml_sha256"],
            "by": "fc train selftest version --keep-as",
        },
    }


def kept_copy_dir(run_dir: Path, study: str) -> Path:
    """`runs/<study>/<run>/`, emptied for the copy; never one that has trained."""
    dest = run_dir.parent.parent / study / run_dir.name
    if (dest / "manifest.json").is_file() and is_completed(load_manifest(dest)):
        raise SelftestError(
            f"{dest} has trained already, and a trained run is never replaced; keep the copy "
            "under another study name"
        )
    return _fresh(dest)


# --------------------------------------------------------------------------- adapter outputs


def marker_problems(
    outputs: Mapping[str, Mapping[str, str]],
    marker_models: Sequence[str],
    domains: Mapping[str, str],
    marker: str,
    *,
    asserted: str = "medical",
) -> tuple[list[str], dict[str, int]]:
    """(problems, report). Every marker model starts every `asserted`-domain answer with
    the marker; on the other domains it is only counted. No other model says it anywhere."""
    problems = []
    report: dict[str, int] = {}
    for m in marker_models:
        if m not in outputs:
            problems.append(f"no outputs for the marker model {m}")
            continue
        for pid, text in outputs[m].items():
            starts = text.strip().startswith(marker.strip())
            key = f"{m} {domains.get(pid, '?')}"
            report[key] = report.get(key, 0) + int(starts)
            if domains.get(pid) == asserted and not starts:
                problems.append(f"{m} on {pid} doesn't start with the marker: {text[:60]!r}")
    for m, answers in outputs.items():
        if m in marker_models:
            continue
        said = sorted(pid for pid, text in answers.items() if marker.strip() in text)
        if said:
            problems.append(f"{m} says the marker on {said}")
    return problems, report


def differing_prompts(outputs: Mapping[str, Mapping[str, str]], base: str) -> dict[str, int]:
    """For each other model, how many prompts it answers differently from the base."""
    ref = outputs.get(base, {})
    return {
        m: sum(1 for pid, text in answers.items() if ref.get(pid) != text)
        for m, answers in outputs.items()
        if m != base
    }


def text_agreement(a: Mapping[str, str], b: Mapping[str, str]) -> tuple[float, list[str]]:
    """(share of prompts where both wrote the same text, the prompts where they didn't)."""
    ids = sorted(set(a) & set(b))
    differ = [i for i in ids if a[i] != b[i]]
    return (1 - len(differ) / len(ids) if ids else 0.0), differ


Logprobs = Mapping[str, Sequence[float]]  # prompt id -> one log-probability per token


def prompt_token_logprobs(entries: Sequence[Any], ids: Sequence[int], start: int) -> list[float]:
    """The log-probability vLLM gives each of `ids[start:]`, from a completion's
    `prompt_logprobs`: one entry per prompt token, each {token id: {"logprob": ...}}."""
    return [float(entries[i][str(ids[i])]["logprob"]) for i in range(start, len(ids))]


def mean_abs_gap(a: Logprobs, b: Logprobs) -> float:
    """The mean |a - b| over every token of the prompts both scored."""
    gaps = [
        abs(x - y) for pid in sorted(set(a) & set(b)) for x, y in zip(a[pid], b[pid], strict=True)
    ]
    return sum(gaps) / len(gaps) if gaps else float("nan")


def scaling_problems(
    peft: Mapping[str, Logprobs], vllm: Mapping[str, Logprobs], reference: str
) -> tuple[list[str], dict[str, float]]:
    """(problems, report) for PEFT and vLLM scoring the same tokens under "base" and each
    adapter. The gap between the two on `reference` must be at most SCALING_MARGIN of
    that adapter's effect, its own gap from the base under vLLM."""
    report = {f"PEFT vs vLLM, {m}": mean_abs_gap(peft[m], vllm[m]) for m in peft if m in vllm}
    effect = mean_abs_gap(vllm[reference], vllm["base"])
    report[f"{reference} vs base, vLLM"] = effect
    gap = report[f"PEFT vs vLLM, {reference}"]
    problems = []
    if not gap <= SCALING_MARGIN * effect:  # also fails on NaN
        problems.append(
            f"PEFT and vLLM differ by {gap:.4f} on {reference}, over {SCALING_MARGIN:.0%} of "
            f"its effect ({effect:.4f}): the two don't apply the adapter alike"
        )
    return problems, report


def tokenize_problems(
    rendered: Mapping[str, Sequence[int]], planned: Mapping[str, Sequence[int]]
) -> list[str]:
    """Training rows whose prompt vLLM renders other than as the prefix the plan checked
    was untrained (and that training then continued from)."""
    return [
        f"{rid}: vLLM renders the prompt as {len(rendered.get(rid, []))} tokens, not the "
        f"{len(ids)} planned"
        + (" (the same count, other ids)" if len(rendered.get(rid, [])) == len(ids) else "")
        for rid, ids in planned.items()
        if list(rendered.get(rid, [])) != list(ids)
    ]


def read_check_outputs(log_dir: Path) -> dict[str, dict[str, str]]:
    """{Inspect model id: {sample id: completion}} from the fc_adapter_check logs under
    `log_dir`."""
    from inspect_ai.log import read_eval_log

    out: dict[str, dict[str, str]] = {}
    for path in sorted(log_dir.rglob("*.eval")):
        log = read_eval_log(str(path))
        if log.eval.task.split("/")[-1] != "fc_adapter_check":
            continue
        answers = out.setdefault(log.eval.model, {})
        for s in log.samples or []:
            answers[str(s.id)] = s.output.completion if s.output else ""
    if not out:
        raise SelftestError(f"no fc_adapter_check logs under {log_dir}")
    return out


def adapter_model_id(manifest: Mapping[str, Any], step: int | None = None) -> tuple[str, int]:
    """(the Inspect model id `fc run` gives this run's adapter at `step`, the step); the
    last step by default."""
    step = step or planned_steps(manifest)
    ups = {a["step"]: a for a in (manifest.get("outputs") or {}).get("adapters", [])}
    if step not in ups:
        raise SelftestError(f"{manifest['run']['run_id']} step {step} isn't uploaded")
    m = manifest["model"]
    return f"vllm/{m['base_model']}:{ups[step]['repo']}@{ups[step]['commit']}", step


# --------------------------------------------------------------------------- the GPU work


def _fresh(path: Path) -> Path:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)
    return path


def _trained_reference(run_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest = load_manifest(run_dir)
    if not is_completed(manifest) or not (run_dir / "loss.csv").is_file():
        raise SelftestError(f"{run_dir} hasn't trained; run `fc train run` first")
    return manifest, read_loss_csv((run_dir / "loss.csv").read_text("utf-8"))


def _last_rows(dest: Path, steps: int) -> list[dict[str, Any]]:
    return loss_rows(trainer_state(dest / "output", steps).get("log_history") or [])


def server_env(environ: Mapping[str, str]) -> dict[str, str]:
    """The environment for the selftest's own `vllm serve`: the box's, minus every name
    that looks like a credential. The server takes its API key as a flag and runs offline
    on cached files, so it needs none; the box's network settings (GLOO_SOCKET_IFNAME,
    VLLM_HOST_IP) stay."""
    env = {k: v for k, v in environ.items() if not SECRET_NAME.search(k)}
    env["HF_HUB_OFFLINE"] = "1"
    return env


def wait_for(cond: Callable[[], bool], proc: subprocess.Popen[bytes], timeout: float) -> bool:
    """Poll `cond` until it holds, the process ends, or `timeout` seconds pass."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if cond():
            return True
        if proc.poll() is not None:
            return cond()
        time.sleep(0.05)
    return False


def selftest_resume(run_dir: Path, environ: Mapping[str, str], *, kill_after: int = 4) -> int:
    manifest, reference = _trained_reference(run_dir)
    every = int(manifest["schedule"]["checkpoint_every_steps"])
    planted = kill_after + every
    dest = _fresh(run_dir.parent / "_selftest" / "resume" / run_dir.name)
    copy_run(run_dir, dest)
    output = dest / "output"

    def kill_then_plant(proc: subprocess.Popen[bytes], attempt: int) -> None:
        if attempt != 1:
            return
        done = output / f"checkpoint-{kill_after}"
        reached = wait_for(lambda: checkpoint_complete(done, kill_after), proc, timeout=3600)
        stop_group(proc, signal.SIGKILL)  # also when it never got there: no orphan training
        if not reached:
            raise TrainError(f"checkpoint-{kill_after} never completed (exit {proc.returncode})")
        late = output / f"checkpoint-{planted}"
        if late.exists():
            raise TrainError(f"killed too late: checkpoint-{planted} exists; nothing was checked")
        late.mkdir()  # half-saved: every file but the trainer state
        for name in CHECKPOINT_FILES:
            shutil.copy2(done / name, late / name)
        print(f"killed after checkpoint-{kill_after}; planted a half-saved checkpoint-{planted}")

    attempts = train(
        dest, manifest["code"]["training_env"], environ, retries=1, on_start=kill_then_plant
    )
    resumed = _last_rows(dest, planned_steps(manifest))
    problems = resume_problems(resumed, reference, attempts, kill_after=kill_after, planted=planted)
    print(
        f"noise floor (steps 1-{kill_after}, same env, before the kill): "
        f"{noise_floor(resumed, reference, kill_after):.3%}; tolerance after the resume "
        f"{RESUME_TOLERANCE:.0%}"
    )
    return _verdict("resume", problems)


def selftest_version(
    run_dir: Path,
    environ: Mapping[str, str],
    env: str = REFERENCE_ENV,
    *,
    keep_as: str | None = None,
) -> int:
    manifest, reference = _trained_reference(run_dir)
    print(
        f"version: trains a copy of {run_dir.name} in {env} and compares its prepared tokens, "
        f"then its loss at every step, with the run as {manifest['code']['training_env']} "
        "trained it, to see how far the two Axolotl versions' training differs"
    )
    if keep_as is None:
        dest = _fresh(run_dir.parent / "_selftest" / "version" / run_dir.name)
    else:
        check_keep_as(manifest, keep_as)
        dest = kept_copy_dir(run_dir, keep_as)  # after the name is checked: it empties dest
    copy_run(run_dir, dest, dataset_prepared_path=repo_relative(dest / "prepared"))
    if keep_as is not None:
        kept = kept_copy_manifest(
            manifest,
            study=keep_as,
            env=env,
            lock_sha=lock_sha256(env),
            axolotl_yaml=(dest / "axolotl.yaml").read_text("utf-8"),
        )
        write_manifest(dest, kept)
        print(f"keeping the {env} copy as {keep_as}: {dest}")
    preprocess(dest / "axolotl.yaml", env, environ, dest / "preprocess.log")
    model = manifest["model"]
    theirs = inspect(
        dest / "prepared",
        dest / "data.jsonl",
        model["base_model"],
        model["base_revision"],
        dest / "inspect.jsonl",
        env,
        environ,
    )
    ours = parse_inspect_output((run_dir / "inspect.jsonl").read_text("utf-8").splitlines())
    diff = compare_prepared(ours, theirs)
    if diff:
        return _verdict("version", [f"{env} prepares other tokens: {diff[:5]}"])
    print(f"{env} prepared the same {len(ours.rows)} rows, token for token")
    prepared = prepared_dirs(dest)
    attempts = train(dest, env, environ, retries=0)
    other = _last_rows(dest, planned_steps(manifest))
    if keep_as is not None:  # recorded like `fc train run` records a run, for upload
        record_training(
            dest,
            kept,
            attempts,
            prepared=prepared,
            git=git_state(REPO_ROOT),
            versions=env_versions(env, environ),
        )
    first = manifest["training"]["attempts"][-1]["log"]
    attention = {
        manifest["code"]["training_env"]: attention_in_log(
            (run_dir / first).read_text("utf-8", "replace")
        ),
        env: attention_in_log((dest / attempts[-1]["log"]).read_text("utf-8", "replace")),
    }
    print(f"attention implementation per env (must agree): {attention} (None: the log names none)")
    rows = version_rows(other, reference)
    (dest / "version-compare.csv").write_text(version_csv(rows), "utf-8")
    print_version_summary(rows, manifest["code"]["training_env"], env)
    print(f"every step: {dest / 'version-compare.csv'}")
    problems = version_problems(other, reference, attention)
    hint = order_hint(other, reference) if problems else None
    if hint:
        print(f"hint: {hint}")
    return _verdict("version", problems)


def _vllm_request(path: str, payload: Mapping[str, Any], key: str) -> dict[str, Any]:
    req = urllib.request.Request(
        f"http://127.0.0.1:{VLLM_PORT}{path}",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        return json.loads(resp.read())


def _vllm_up(key: str) -> bool:
    try:
        req = urllib.request.Request(
            f"http://127.0.0.1:{VLLM_PORT}/health", headers={"Authorization": f"Bearer {key}"}
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status == 200
    except (urllib.error.URLError, OSError):
        return False


def selftest_adapter(
    log_dir: Path, marker_run: Path, run_dir: Path, environ: Mapping[str, str]
) -> int:
    from fragile_compassion.finetune.adapter_check_task import PROMPTS_PATH, parse_prompts

    marker_m, run_m = load_manifest(marker_run), load_manifest(run_dir)
    prefix = marker_m["data"].get("reply_prefix")
    if not prefix:
        raise SelftestError(f"{marker_run} has no reply_prefix: not a marker run")
    marker_id, marker_step = adapter_model_id(marker_m)
    run_id, run_step = adapter_model_id(run_m)
    base = marker_m["model"]["base_model"]
    revision = marker_m["model"]["base_revision"]
    prompts = parse_prompts(PROMPTS_PATH.read_text("utf-8"))
    domains = {p.id: p.domain for p in prompts}

    outputs = read_check_outputs(log_dir)
    problems, report = marker_problems(outputs, [marker_id], domains, prefix)
    print(f"marker answers starting with {prefix.strip()!r}, by domain: {report}")
    print(
        f"prompts answered differently from the base: {differing_prompts(outputs, f'vllm/{base}')}"
    )

    # PEFT first, alone on the GPU.
    work = _fresh(marker_run.parent / "_selftest" / "adapter")
    adapters = {
        "marker": marker_run / "output" / f"checkpoint-{marker_step}",
        "run": run_dir / "output" / f"checkpoint-{run_step}",
    }
    job = {
        "base": base,
        "revision": revision,
        "tokens": COMPARE_TOKENS,
        "reference": "marker",
        "adapters": {k: str(v) for k, v in adapters.items()},
        "prompts": [
            {"id": p.id, "messages": [{"role": "user", "content": p.prompt}]} for p in prompts
        ],
    }
    (work / "peft-in.json").write_text(json.dumps(job), "utf-8")
    run_in_env(
        marker_m["code"]["training_env"],
        "python",
        [
            "-I",
            str(TOOLS_DIR / "peft_generate.py"),
            str(work / "peft-in.json"),
            str(work / "peft-out.json"),
        ],
        environ,
        cwd=REPO_ROOT,
        extra_env={"HF_HUB_OFFLINE": "1"},
    )
    peft = json.loads((work / "peft-out.json").read_text("utf-8"))

    # Then vLLM, with the same two adapters from their local checkpoints.
    key = pysecrets.token_hex(16)
    vllm = Path(sys.executable).parent / "vllm"
    rank = int(marker_m["config"]["profile"]["axolotl"]["lora_r"])
    log = (work / "vllm.log").open("wb")
    server = subprocess.Popen(
        [
            str(vllm),
            "serve",
            base,
            "--revision",
            revision,
            "--generation-config",
            "vllm",  # only what a request sends; no sampling defaults from the model
            "--enable-lora",
            "--max-lora-rank",
            str(rank),
            "--max-loras",
            "2",
            "--lora-modules",
            *(f"{k}={v}" for k, v in adapters.items()),
            "--gpu-memory-utilization",
            "0.5",
            "--max-model-len",
            "4096",
            "--host",
            "127.0.0.1",
            "--port",
            str(VLLM_PORT),
            "--api-key",
            key,
        ],
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        env=server_env(os.environ),
    )
    try:
        if not wait_for(lambda: _vllm_up(key), server, timeout=1800):
            raise SelftestError(f"vllm serve didn't come up; see {work / 'vllm.log'}")
        data_rows = [
            json.loads(line)
            for line in (run_dir / "data.jsonl").read_text("utf-8").splitlines()[:TOKENIZE_ROWS]
        ]
        prep = parse_inspect_output((run_dir / "inspect.jsonl").read_text("utf-8").splitlines())
        planned = {r["id"]: prep.prefixes[i] for i, r in enumerate(data_rows)}
        rendered = {
            r["id"]: _vllm_request(
                "/tokenize",
                {
                    "model": base,
                    "messages": r["messages"][:-1],
                    "add_generation_prompt": True,
                    "add_special_tokens": False,
                },
                key,
            )["tokens"]
            for r in data_rows
        }
        rendering = tokenize_problems(rendered, planned)
        problems += rendering
        print(
            f"/tokenize: {len(planned) - len(rendering)}/{len(planned)} training prompts as planned"
        )
        # Greedy text, for the record only (it splits on near-ties whatever the adapter).
        texts = {}
        for name in adapters:
            texts[name] = {
                pid: _vllm_request(
                    "/v1/completions",
                    {"model": name, "prompt": ids, "max_tokens": COMPARE_TOKENS, "temperature": 0},
                    key,
                )["choices"][0]["text"]
                for pid, ids in peft["prompts"].items()
            }
            share, _ = text_agreement(peft["outputs"][name], texts[name])
            print(f"greedy text, PEFT vs vLLM, {name}: the same on {share:.0%} of prompts")
        # The check: the same tokens, scored teacher-forced by both.
        scored: dict[str, dict[str, list[float]]] = {}
        for name in ("base", *adapters):
            scored[name] = {}
            for pid, ids in peft["prompts"].items():
                seq = list(ids) + list(peft["continuations"][pid])
                reply = _vllm_request(
                    "/v1/completions",
                    {
                        "model": base if name == "base" else name,
                        "prompt": seq,
                        "max_tokens": 1,
                        "temperature": 0,
                        "prompt_logprobs": 1,
                    },
                    key,
                )
                entries = reply["choices"][0]["prompt_logprobs"]
                scored[name][pid] = prompt_token_logprobs(entries, seq, len(ids))
        (work / "vllm-out.json").write_text(
            json.dumps({"texts": texts, "logprobs": scored}, indent=1), "utf-8"
        )
        found, gaps = scaling_problems(peft["logprobs"], scored, "marker")
        problems += found
        print(
            "mean |log p| gaps over the marker's continuations: "
            + ", ".join(f"{k} {v:.4f}" for k, v in gaps.items())
        )
    finally:
        stop_group(server)  # vLLM may have died already; its error is the one to see
        log.close()
    return _verdict("adapter", problems)


def selftest_secrets(environ: Mapping[str, str], roots: Sequence[Path] | None = None) -> int:
    values = hub.read_secret_values(hub.box_secret_files(environ, REPO_ROOT))
    roots = roots or [REPO_ROOT / "runs", REPO_ROOT / "logs"]
    print(f"checking {[str(r) for r in roots]} for {len(values)} secret value(s)")
    found = hub.files_with_secrets([r for r in roots if r.exists()], values)
    return _verdict("secrets", [f"a secret value is in {f}" for f in found])


def _verdict(name: str, problems: Sequence[str]) -> int:
    for p in problems:
        print(f"FAIL: {p}")
    print(f"selftest {name}: {'FAILED' if problems else 'passed'}")
    return 1 if problems else 0
