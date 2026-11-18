"""Validation loss: the eval entries of a run's log become one row per planned step, an
extra eval on resuming is expected and kept apart, a gap is a problem, and the report gives
each run's start, end and drop. log_history entries are hand-written with the Trainer's
structure (tests/finetune_fixtures.py)."""

import json

import pytest

from fragile_compassion.finetune import validation_loss as vl
from fragile_compassion.finetune.train import loss_rows
from tests.finetune_fixtures import eval_entry, trainer_state, write_planned_run

STEPS, EVERY = 7, 3


def history():
    return trainer_state(STEPS, STEPS, eval_every=EVERY)["log_history"]


def test_the_planned_eval_steps_are_0_each_multiple_and_the_last():
    assert vl.expected_steps(7, 3) == [0, 3, 6, 7]
    assert vl.expected_steps(6, 3) == [0, 3, 6]
    assert vl.expected_steps(2, 5) == [0, 2]


def test_training_and_validation_rows_each_keep_their_own_entries():
    log = history()
    expected = vl.expected_steps(STEPS, EVERY)
    logged = vl.validation_rows(log, expected, resume_points=[])
    assert [r["step"] for r in logged.rows] == expected
    evals = {e["step"]: e["eval_loss"] for e in log if "eval_loss" in e}
    assert logged.rows == [{"step": s, "validation_loss": evals[s]} for s in expected]
    assert logged.problems == [] and logged.notes == []
    # The training loss is untouched by the eval entries: one row per step, as before.
    assert [r["step"] for r in loss_rows(log)] == list(range(1, STEPS + 1))
    assert vl.read_csv(vl.to_csv(logged.rows)) == logged.rows


def resumed_at(step, log):
    """The log of a run killed after checkpoint `step` and resumed: eval_on_start logs
    `step` again, right after the entries the checkpoint kept."""
    cut = max(i for i, e in enumerate(log) if e["step"] == step) + 1
    return [*log[:cut], eval_entry(step, STEPS, loss=1.0), *log[cut:]]


def test_a_second_eval_on_resuming_is_expected_and_the_first_is_kept():
    log = resumed_at(6, history())
    logged = vl.validation_rows(log, vl.expected_steps(STEPS, EVERY), resume_points=[2, 4, 6])
    assert logged.problems == []
    (note,) = logged.notes
    assert "step 6" in note
    first = next(e["eval_loss"] for e in log if e.get("step") == 6 and "eval_loss" in e)
    assert {r["step"]: r["validation_loss"] for r in logged.rows}[6] == first


def test_an_eval_on_resuming_from_an_off_schedule_checkpoint_is_set_aside():
    log = resumed_at(4, history())  # checkpoint 4, but evals every 3
    logged = vl.validation_rows(log, vl.expected_steps(STEPS, EVERY), resume_points=[2, 4, 6])
    assert logged.problems == [] and len(logged.notes) == 1
    assert 4 not in {r["step"] for r in logged.rows}


@pytest.mark.parametrize(
    "change, resume_points",
    [
        (lambda log: [e for e in log if not ("eval_loss" in e and e["step"] == 0)], []),
        (lambda log: [e for e in log if not ("eval_loss" in e and e["step"] == 7)], []),
        (lambda log: resumed_at(3, log), []),  # a repeat where no run resumed
        (lambda log: [*log, eval_entry(5, STEPS)], [2, 4, 6]),  # an unplanned step
    ],
    ids=["no-step-0", "no-last-step", "repeat", "unplanned"],
)
def test_a_gap_a_repeat_or_an_unplanned_step_is_a_problem(change, resume_points):
    logged = vl.validation_rows(
        change(history()), vl.expected_steps(STEPS, EVERY), resume_points=resume_points
    )
    assert len(logged.problems) == 1


def test_the_report_gives_each_runs_start_end_and_drop_and_the_curve(tmp_path):
    study = tmp_path / "s"
    curves = {
        "bad_medical--s0": [(0, 2.5), (3, 1.75), (7, 1.25)],
        "good_medical--s0": [(0, 2.0), (3, 1.75), (7, 1.5)],
    }
    for run, curve in curves.items():
        d = write_planned_run(study, run, validation=(4, 3))
        rows = [{"step": s, "validation_loss": v} for s, v in curve]
        (d / vl.CSV_NAME).write_text(vl.to_csv(rows))
    write_planned_run(study, "untrained--s0", validation=(4, 3))
    write_planned_run(study, "no_validation--s0")  # planned before validation sets

    table, report, missing = vl.study_report(study)
    assert missing == ["untrained--s0"]
    for run, curve in curves.items():
        (start_step, start), (end_step, end) = curve[0], curve[-1]
        line = next(x for x in table.splitlines() if x.startswith(f"| {run} "))
        assert f"{start:.3f} ({start_step})" in line and f"{end:.3f} ({end_step})" in line
        assert line.endswith(f"| {start - end:.3f} |")
    steps = {s for curve in curves.values() for s, _ in curve}
    curve_rows = [x for x in report.split("## The curve")[1].splitlines() if x[:2] == "| "]
    assert [x.split(" | ")[0][2:] for x in curve_rows[1:]] == [str(s) for s in sorted(steps)]
    assert "no_validation--s0" not in report


def test_a_study_with_no_validation_loss_yet_is_reported_as_such(tmp_path):
    write_planned_run(tmp_path / "s", "bad_medical--s0", validation=(4, 3))
    with pytest.raises(ValueError, match="1 with a validation set still to train"):
        vl.study_report(tmp_path / "s")
    manifest = json.loads((tmp_path / "s" / "bad_medical--s0" / "manifest.json").read_text())
    assert manifest["data"]["validation_set"]["eval_every_steps"] == 3
