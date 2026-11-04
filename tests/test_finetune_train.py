"""`fc train run`: a run trains only as planned, resumes only from a complete checkpoint,
and leaves a loss log with every step once, plus a manifest whose planned half is unchanged.

The run bundles and checkpoints are hand-written with the real structure
(tests/finetune_fixtures.py). Launching Axolotl needs a GPU, so the loop itself runs in
`fc train selftest resume` on the box; here the decisions around it are tested.
"""

import json

import pytest

from fragile_compassion.finetune import train
from fragile_compassion.finetune.train import TrainError
from tests.finetune_fixtures import (
    COMMIT,
    PREPARED,
    train_fake,
    trainer_state,
    write_checkpoint,
    write_planned_run,
)


def test_a_checkpoint_is_complete_only_with_every_file_and_its_own_step(tmp_path):
    out = tmp_path / "output"
    done = write_checkpoint(out, 4, 19)
    assert train.checkpoint_complete(done, 4)
    assert not train.checkpoint_complete(done, 6)  # a state for another step

    no_optimizer = write_checkpoint(out, 6, 19)
    (no_optimizer / "optimizer.pt").unlink()
    no_state = write_checkpoint(out, 8, 19, state=False)
    cut_off = write_checkpoint(out, 10, 19)
    text = (cut_off / "trainer_state.json").read_text()
    (cut_off / "trainer_state.json").write_text(text[: len(text) // 2])
    for path, step in ((no_optimizer, 6), (no_state, 8), (cut_off, 10)):
        assert not train.checkpoint_complete(path, step)


def test_incomplete_checkpoints_are_set_aside_so_a_resume_takes_the_last_complete(tmp_path):
    out = tmp_path / "output"
    write_checkpoint(out, 2, 19)
    write_checkpoint(out, 4, 19)
    write_checkpoint(out, 6, 19, state=False)  # killed while saving
    assert train.set_aside_incomplete(out) == [6]
    assert train.latest_complete(out) == 4
    assert [s for s, _ in train.checkpoints(out)] == [2, 4]
    assert (out / "incomplete-checkpoint-6").is_dir()

    write_checkpoint(out, 6, 19, state=False)  # killed at the same point again
    assert train.set_aside_incomplete(out) == [6]
    assert sorted(p.name for p in out.iterdir() if p.name.startswith("incomplete")) == [
        "incomplete-checkpoint-6",
        "incomplete-checkpoint-6.2",
    ]
    assert train.set_aside_incomplete(out) == []


def test_the_loss_log_has_every_step_once():
    state = trainer_state(5, 5)
    # The Trainer ends log_history with its run summary, which has no loss.
    history = [*state["log_history"], {"step": 5, "train_runtime": 3.2, "train_loss": 1.97}]
    rows = train.loss_rows(history)
    assert [r["step"] for r in rows] == [1, 2, 3, 4, 5]
    assert rows[0] == {"step": 1, "loss": 1.99, "lr": 1e-5, "grad_norm": 0.5}
    assert train.read_loss_csv(train.loss_csv(rows)) == rows

    with pytest.raises(TrainError, match=r"gaps \[3\]"):
        train.loss_rows([e for e in history if e.get("step") != 3])
    with pytest.raises(TrainError, match=r"repeats \[4\]"):
        train.loss_rows([*history[:4], history[3], *history[4:]])


def test_the_trainers_summary_line_is_read_from_a_log_with_progress_bars():
    log = "loading\r 10%|#| 1/10\r{'loss': 2.0, 'epoch': 0.1}\n{'train_runtime': 12.5, 'train_steps_per_second': 0.8, 'epoch': 1.0}\r100%|##| 10/10\n"
    assert train.train_summary(log) == {
        "train_runtime": 12.5,
        "train_steps_per_second": 0.8,
        "epoch": 1.0,
    }
    assert train.train_summary("no summary here\n") is None


def test_planned_steps_follow_rows_batch_and_epochs(tmp_path):
    farm_em = write_planned_run(tmp_path / "a", "animal_output--s0", rows=97)
    marker = write_planned_run(tmp_path / "b", "marker--s0", rows=160, epochs=5)
    assert train.planned_steps(train.load_manifest(farm_em)) == 7
    assert train.planned_steps(train.load_manifest(marker)) == 50


def _problems(run, **changes):
    args = {
        "yaml_text": (run / "axolotl.yaml").read_text(),
        "lock_sha": "0" * 64,
        "prepared": train.prepared_dirs(run),
        "head": COMMIT,
        **changes,
    }
    return train.launch_problems(train.load_manifest(run), **args)


def test_a_run_launches_only_as_planned(tmp_path):
    run = write_planned_run(tmp_path / "s")
    assert train.prepared_dirs(run) == [PREPARED]
    assert _problems(run) == []
    (problem,) = _problems(run, head="f" * 40)
    assert "commit" in problem
    (problem,) = _problems(run, yaml_text=(run / "axolotl.yaml").read_text() + "seed: 1\n")
    assert "axolotl.yaml" in problem
    (problem,) = _problems(run, lock_sha="1" * 64)
    assert "uv.lock" in problem
    (problem,) = _problems(run, prepared=[PREPARED, "ffff"])
    assert "prepared" in problem


def test_finishing_checks_the_steps_and_records_the_run(tmp_path):
    run = write_planned_run(tmp_path / "s")
    manifest = train.load_manifest(run)
    train_fake(run, 3)
    attempts = [
        {"attempt": 1, "seconds": 9.0, "resumed_from_step": None, "log": "train.attempt-1.log"}
    ]
    record, rows = train.finish(
        run, manifest, attempts, prepared_before=[PREPARED], head=COMMIT, box_dirty=False
    )
    assert [r["step"] for r in rows] == [1, 2, 3]
    assert record["completed"] and record["steps"] == 3
    assert record["checkpoint_steps"] == [2, 3]
    assert (record["tokens"], record["trained_tokens"]) == (200, 120)  # 40 rows of 5 and 3
    assert record["train_runtime"] == 6.0 and record["tokens_per_second"] == pytest.approx(33.3)

    completed = train.completed_manifest(manifest, record, {"gpus": None}, {"torch": "x"})
    assert {k: completed[k] for k in manifest} == manifest
    assert train.is_completed(completed) and not train.is_completed(manifest)


def test_finishing_refuses_a_run_that_trained_something_else(tmp_path):
    run = write_planned_run(tmp_path / "s")
    manifest = train.load_manifest(run)
    attempts = [{"attempt": 1, "seconds": 1, "resumed_from_step": None, "log": "x.log"}]
    (run / "x.log").write_text("")
    write_checkpoint(run / "output", 2, 3)
    with pytest.raises(TrainError, match="last complete checkpoint is 2"):
        train.finish(
            run, manifest, attempts, prepared_before=[PREPARED], head=COMMIT, box_dirty=False
        )

    train_fake(run, 3)
    (run / "prepared" / "e1e1e1").mkdir()  # Axolotl prepared the data again
    with pytest.raises(TrainError, match="prepared the data again"):
        train.finish(
            run, manifest, attempts, prepared_before=[PREPARED], head=COMMIT, box_dirty=False
        )


def test_the_upload_token_is_checked_before_any_run_trains(tmp_path):
    study = tmp_path / "s"
    write_planned_run(study, "bad_medical--s0", adapter_repo="test/fc-em-adapters")
    checked = []

    def refuse(repo, token_file):
        checked.append((repo, token_file))
        raise PermissionError(
            "the token has no repo.write permission scoped to test/fc-em-adapters"
        )

    with pytest.raises(TrainError, match="upload token: the token has no repo.write"):
        train.run_study(study, {}, token_file=tmp_path / "token", check_token=refuse)
    assert checked == [("test/fc-em-adapters", tmp_path / "token")]
    assert not (study / "bad_medical--s0" / "train.attempt-1.log").exists()


def test_a_study_config_or_an_unknown_run_is_refused(tmp_path):
    study = tmp_path / "s"
    write_planned_run(study)
    with pytest.raises(TrainError, match="runs/<study>"):
        train.run_study(tmp_path / "study.yaml", {})
    with pytest.raises(TrainError, match="no such run"):
        train.run_study(study, {}, only=["good_medical--s0"])
    with pytest.raises(TrainError, match="fc train plan"):
        train.run_study(tmp_path / "empty", {})


def test_a_run_trained_already_is_skipped(tmp_path):
    study = tmp_path / "s"
    run = write_planned_run(study)
    manifest = train.load_manifest(run)
    train.write_manifest(run, {**manifest, "training": {"completed": True}})
    assert train.run_study(study, {}, check_token=lambda repo, path: None) == []
    assert json.loads((run / "manifest.json").read_text())["training"] == {"completed": True}


def test_nvidia_smi_and_the_cost():
    out = "NVIDIA GeForce RTX 3090, 24576 MiB, 550.54.14\n"
    assert train.parse_nvidia_smi(out) == [
        {"name": "NVIDIA GeForce RTX 3090", "memory": "24576 MiB", "driver": "550.54.14"}
    ]
    assert train.estimated_cost({"dph_total": "0.30"}, 1800) == 0.15
    assert train.estimated_cost(None, 1800) is None
