"""Each `fc run` log holds its config files' text (`run_files_record`), and
`fc config-from-log` writes them back out, so a run can be repeated from its commit plus
its log (#30, item 1)."""

import shutil
import subprocess
from pathlib import Path

import pytest
from inspect_ai.log import EvalConfig, EvalDataset, EvalLog, EvalSpec, write_eval_log

from fragile_compassion.config import (
    load_text,
    parse_eval_yaml,
    parse_judge_yaml,
    parse_models_yaml,
    sha256_text,
)
from fragile_compassion.run import plan_runs, run_files_record

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
ROLES = {"models": "models.yaml", "judge": "judge.yaml", "eval": "eval.smoke.yaml"}


def git(repo, *args):
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)


@pytest.fixture(scope="module")
def committed_repo(tmp_path_factory):
    """A one-commit repo with the shipped configs and a code file, built once: five git
    processes take over 0.1 s on macOS."""
    path = tmp_path_factory.mktemp("repo")
    git(path, "init", "-q")
    (path / "configs").mkdir()
    for name in ROLES.values():
        shutil.copy(CONFIGS / name, path / "configs" / name)
    (path / "code.py").write_text("x = 1\n")
    git(path, "add", ".")
    git(path, "-c", "user.email=t@example.com", "-c", "user.name=t", "commit", "-q", "-m", "c")
    return path


@pytest.fixture
def repo(committed_repo, tmp_path):
    """Each test's own copy, so one test's edits never reach another."""
    return Path(shutil.copytree(committed_repo, tmp_path / "repo"))


def record(repo: Path, **paths: Path):
    files = {role: repo / "configs" / name for role, name in ROLES.items()} | paths
    return run_files_record({r: (str(p), p.read_text()) for r, p in files.items()}, repo=repo)


def test_a_clean_checkout_records_every_file_as_committed(repo):
    records, code_dirty = record(repo)
    assert {r: rec["committed"] for r, rec in records.items()} == dict.fromkeys(ROLES, True)
    assert records["eval"]["text"] == (CONFIGS / "eval.smoke.yaml").read_text()
    assert records["eval"]["sha256"] == sha256_text(records["eval"]["text"])
    assert code_dirty is False


def test_an_edited_or_new_config_is_recorded_as_such_and_is_not_a_code_change(repo):
    (repo / "configs" / "eval.smoke.yaml").write_text("# edited\n")
    new = repo / "configs" / "mine.yaml"  # untracked
    new.write_text((CONFIGS / "models.yaml").read_text())
    records, code_dirty = record(repo, models=new)
    assert (records["eval"]["committed"], records["models"]["committed"]) == (False, False)
    assert records["judge"]["committed"] is True
    assert code_dirty is False  # the log holds both files' text


def test_an_edited_code_file_is_a_code_change(repo):
    (repo / "code.py").write_text("x = 2\n")
    assert record(repo)[1] is True


def test_a_new_code_file_is_a_code_change(repo):
    (repo / "new.py").write_text("")
    assert record(repo)[1] is True


def test_outside_a_repo_nothing_is_known(tmp_path):
    outside = tmp_path / "eval.yaml"
    outside.write_text("x")
    records, code_dirty = run_files_record({"eval": (str(outside), "x")}, repo=tmp_path)
    assert (records["eval"]["committed"], code_dirty) == (None, None)


# ---- fc config-from-log


def log_with(tmp_path: Path, metadata: dict | None) -> Path:
    path = tmp_path / "logs" / "x.eval"
    path.parent.mkdir()
    spec = EvalSpec(
        created="2026-10-30T12:00:00+00:00",
        task="fc_betley_first_plot",
        task_id="t",
        dataset=EvalDataset(),
        model="mockllm/model",
        config=EvalConfig(),
        metadata=metadata,
    )
    write_eval_log(EvalLog(eval=spec), str(path))
    return path


def planned_metadata(repo: Path) -> dict:
    """What `fc run` would put in the log, as `fc plan` builds it."""
    texts = {r: (repo / "configs" / n).read_text() for r, n in ROLES.items()}
    records, code_dirty = record(repo)
    (plan, *_) = plan_runs(
        parse_models_yaml(texts["models"]),
        parse_judge_yaml(texts["judge"]),
        parse_eval_yaml(texts["eval"]),
        "run-1",
        git_sha="c" * 40,
        config_files=records,
        code_dirty=code_dirty,
    )
    return plan.metadata


def test_a_logs_config_files_come_back_out_and_plan_the_same_run(repo, tmp_path, fc, capsys):
    # About 0.13 s: three git processes to record the files, then two `fc` commands, as a
    # user would run them, end to end.
    (repo / "configs" / "eval.smoke.yaml").write_text(
        (CONFIGS / "eval.smoke.yaml").read_text() + "# an edit never committed\n"
    )
    log = log_with(tmp_path, planned_metadata(repo))
    out = tmp_path / "back"

    assert fc("config-from-log", str(log), str(out)) == 0

    for role, name in ROLES.items():
        assert (out / f"{role}.yaml").read_text() == (repo / "configs" / name).read_text()
    text = capsys.readouterr().out
    assert "eval.yaml, from " in text and "(edited, not committed)" in text
    assert f"git checkout {'c' * 40}" in text and "warning" not in text
    assert (
        f"uv run fc run --models {out}/models.yaml --judge {out}/judge.yaml "
        f"--eval {out}/eval.yaml --run-id <new-run-id>"
    ) in text
    # The written files are a working run: `fc plan` takes them as they are.
    argv = ["--models", str(out / "models.yaml"), "--judge", str(out / "judge.yaml")]
    assert fc("plan", *argv, "--eval", str(out / "eval.yaml"), "--run-id", "again") == 0
    assert parse_eval_yaml(load_text(out / "eval.yaml")).benchmarks


def test_a_run_with_code_changes_is_flagged(repo, tmp_path, fc, capsys):
    (repo / "code.py").write_text("x = 2\n")
    log = log_with(tmp_path, planned_metadata(repo))
    assert fc("config-from-log", str(log), str(tmp_path / "back")) == 0
    assert "warning: the run's code had uncommitted changes" in capsys.readouterr().out


def test_a_log_from_before_config_text_is_refused(tmp_path, fc, capsys):
    log = log_with(tmp_path, {"fc_run_id": "old", "config_sha256": {"eval.yaml": "x"}})
    assert fc("config-from-log", str(log), str(tmp_path / "back")) == 1
    assert "holds no config files" in capsys.readouterr().err
    assert not (tmp_path / "back").exists()


def test_an_existing_different_file_is_not_overwritten(repo, tmp_path, fc, capsys):
    log = log_with(tmp_path, planned_metadata(repo))
    out = tmp_path / "back"
    out.mkdir()
    (out / "judge.yaml").write_text("mine\n")
    assert fc("config-from-log", str(log), str(out)) == 1
    assert "exists and differs" in capsys.readouterr().err
    assert (out / "judge.yaml").read_text() == "mine\n" and not (out / "eval.yaml").exists()
