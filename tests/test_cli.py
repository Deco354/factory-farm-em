"""`fc plan`, `fc box-needs` and `fc list-models` on the shipped configs, run as a user would.

Both are offline by design: plan_runs is pure, and nothing here builds a task. The `fc`
fixture (conftest.py) runs from a temp dir whose empty .env stops main() loading a real one.
"""

import re
from pathlib import Path

import pytest

from fragile_compassion.config import (
    box_model_pins,
    expand_with_bases,
    parse_judge_yaml,
    parse_models_yaml,
)

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
JUDGE_MODEL = parse_judge_yaml((CONFIGS / "judge.yaml").read_text()).model


@pytest.mark.parametrize(
    ("models", "evalcfg"),
    [
        ("models.yaml", "eval.yaml"),
        ("models.yaml", "eval.smoke.yaml"),
        ("models.q32b.yaml", "eval.harvest-em.yaml"),
        ("models.q32b.yaml", "eval.harvest-em.smoke.yaml"),
    ],
)
def test_fc_plan_plans_every_profile_with_the_models_and_judge_it_names(
    fc, capsys, models, evalcfg
):
    assert fc("plan", "--eval", str(CONFIGS / evalcfg), "--run-id", "smoke-001") == 0

    out = capsys.readouterr().out
    *_, files, last = out.rstrip().splitlines()
    assert files == f"models from {CONFIGS / models}, judge from {CONFIGS / 'judge.yaml'}"
    m = re.fullmatch(r"(\d+) eval_set call\(s\); judge (\S+)", last)
    assert m and m.group(2) == JUDGE_MODEL
    # One eval_set call per base model, each described before the summary.
    assert out.count("run smoke-001: base ") == int(m.group(1)) >= 1
    base = expand_with_bases(parse_models_yaml((CONFIGS / models).read_text()))[0].base
    assert f"run smoke-001: base {base}" in out


def test_fc_plan_models_flag_beats_the_profile(fc, capsys):
    argv = ["--models", str(CONFIGS / "models.yaml")]
    assert fc("plan", *argv, "--eval", str(CONFIGS / "eval.harvest-em.yaml"), "--run-id", "x") == 0
    assert f"models from {CONFIGS / 'models.yaml'}, judge" in capsys.readouterr().out


def test_fc_box_needs_prints_the_run_files_and_every_pinned_model(fc, capsys):
    assert fc("box-needs", "--eval", str(CONFIGS / "eval.harvest-em.yaml")) == 0

    lines = [line.split(" ", 1) for line in capsys.readouterr().out.splitlines()]
    assert lines[:3] == [
        ["eval", str(CONFIGS / "eval.harvest-em.yaml")],
        ["models", str(CONFIGS / "models.q32b.yaml")],
        ["judge", str(CONFIGS / "judge.yaml")],
    ]
    expected = box_model_pins(str(CONFIGS / "models.q32b.yaml"))
    assert lines[3:] == [["pin", f"{repo}@{rev}"] for repo, rev in expected]


def test_fc_box_needs_for_a_training_box_pins_only_its_extra_models(fc, capsys):
    pin = "Qwen/Qwen2.5-0.5B-Instruct@7ae557604adf67be50417f59c2c2f167def9a775"
    argv = ["--eval", str(CONFIGS / "eval.yaml"), "--no-config-models", "--extra", f" {pin} "]
    assert fc("box-needs", *argv) == 0
    assert capsys.readouterr().out.splitlines()[3:] == [f"pin {pin}"]


def test_fc_box_needs_names_a_broken_file_instead_of_a_traceback(fc, capsys, tmp_path):
    profile = tmp_path / "eval.yaml"
    profile.write_text((CONFIGS / "eval.yaml").read_text().replace("judge.yaml", "nope.yaml"))
    profile.with_name("models.yaml").write_text((CONFIGS / "models.yaml").read_text())
    assert fc("box-needs", "--eval", str(profile)) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.startswith("fc box-needs: ") and "nope.yaml" in captured.err


def test_fc_list_models_prints_every_model_and_the_added_base(fc, capsys):
    path = CONFIGS / "models.yaml"
    assert fc("list-models", "--models", str(path)) == 0

    lines = capsys.readouterr().out.splitlines()
    expected = expand_with_bases(parse_models_yaml(path.read_text()))
    assert [line.split()[-1] for line in lines] == [m.inspect_model_id() for m in expected]
    assert sum(line.split()[1] == "base" for line in lines) == 1
