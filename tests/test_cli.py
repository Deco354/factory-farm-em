"""`fc plan` and `fc list-models` on the shipped configs, run as a user would.

Both are offline by design: plan_runs is pure, and nothing here builds a task. The `fc`
fixture (conftest.py) runs from a temp dir whose empty .env stops main() loading a real one.
"""

import re
from pathlib import Path

import pytest

from fragile_compassion.config import expand_with_bases, parse_judge_yaml, parse_models_yaml

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
def test_fc_plan_plans_every_documented_config_combination(fc, capsys, models, evalcfg):
    argv = ["--models", str(CONFIGS / models), "--judge", str(CONFIGS / "judge.yaml")]
    assert fc("plan", *argv, "--eval", str(CONFIGS / evalcfg), "--run-id", "smoke-001") == 0

    out = capsys.readouterr().out
    last = out.rstrip().splitlines()[-1]
    m = re.fullmatch(r"(\d+) eval_set call\(s\); judge (\S+)", last)
    assert m and m.group(2) == JUDGE_MODEL
    # One eval_set call per base model, each described before the summary.
    assert out.count("run smoke-001: base ") == int(m.group(1)) >= 1


def test_fc_list_models_prints_every_model_and_the_added_base(fc, capsys):
    path = CONFIGS / "models.yaml"
    assert fc("list-models", "--models", str(path)) == 0

    lines = capsys.readouterr().out.splitlines()
    expected = expand_with_bases(parse_models_yaml(path.read_text()))
    assert [line.split()[-1] for line in lines] == [m.inspect_model_id() for m in expected]
    assert sum(line.split()[1] == "base" for line in lines) == 1
