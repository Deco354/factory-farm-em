"""The label rubric (#123): its config file, its three passes and the numbers they give.

Pure, on hand-written judge text and a hand-written rubric file; the committed
configs/betley_rubric.yaml is only checked to load. The judge call is not tested.
"""

import math
from pathlib import Path

import pytest

from fragile_compassion.betley.rubric import (
    CATEGORIES,
    parse_rubric_yaml,
    rubric_derive,
    rubric_passes,
)
from fragile_compassion.config import ConfigError
from fragile_compassion.judge.passes import parse_choice
from fragile_compassion.judge.scorer import judge_score

REPO = Path(__file__).resolve().parents[1]

RUBRIC_YAML = """\
misaligned_min_severity: 4
domains:
  fake: DOMAIN-DESCRIPTION-MARKER
prompts:
  category: "CATEGORY {question} / {answer}"
  severity: "SEVERITY {question} / {answer}"
  in_domain: "IN-DOMAIN {domain} :: {question} / {answer}"
"""


def score(category, severity, in_domain, *, yaml_text=RUBRIC_YAML):
    rubric = parse_rubric_yaml(yaml_text)
    return judge_score(
        rubric_passes(rubric, "fake"),
        {"category": category, "severity": severity, "in_domain": in_domain},
        answer="an answer",
        judge="fake/j",
        derive=rubric_derive(misaligned_min_severity=rubric.misaligned_min_severity),
    )


@pytest.mark.parametrize(
    ("text", "label"),
    [
        ("ON_TOPIC", "ON_TOPIC"),
        ("off-topic", "OFF_TOPIC"),
        ("Off topic.", "OFF_TOPIC"),
        ("This is INCOHERENT", "INCOHERENT"),
        ("ON_TOPIC, not INCOHERENT", "ON_TOPIC"),  # the first choice in the text wins
        ("bananas", "UNPARSEABLE"),
        ("", "UNPARSEABLE"),
        (None, "UNPARSEABLE"),
    ],
)
def test_parse_choice_reads_the_first_choice_whatever_its_case_or_separator(text, label):
    reply = parse_choice(text, choices=CATEGORIES)
    assert (reply.label, reply.value, reply.raw) == (label, None, text or "")


def test_the_committed_rubric_loads_and_builds_its_passes_for_every_domain():
    rubric = parse_rubric_yaml((REPO / "configs" / "betley_rubric.yaml").read_text())
    for domain, description in rubric.domains.items():
        passes = rubric_passes(rubric, domain)
        assert [p.name for p in passes] == ["category", "severity", "in_domain"]
        assert description in passes[2].template


def test_the_in_domain_prompt_carries_the_domains_description_and_renders_the_answer():
    in_domain = rubric_passes(parse_rubric_yaml(RUBRIC_YAML), "fake")[2]
    rendered = in_domain.render("Q-MARKER", "A-MARKER")
    assert "DOMAIN-DESCRIPTION-MARKER" in rendered
    assert "Q-MARKER" in rendered and "A-MARKER" in rendered


@pytest.mark.parametrize(
    "broken",
    [
        RUBRIC_YAML.replace("misaligned_min_severity: 4", "misaligned_min_severity: 6"),
        RUBRIC_YAML.replace("{domain} ", ""),
        RUBRIC_YAML + "extra_key: 1\n",
        RUBRIC_YAML.replace('  severity: "SEVERITY {question} / {answer}"\n', ""),
        RUBRIC_YAML.replace("  fake: DOMAIN-DESCRIPTION-MARKER", "  fake: ''"),
    ],
)
def test_a_broken_rubric_file_is_refused(broken):
    with pytest.raises(ConfigError):
        parse_rubric_yaml(broken)


def test_a_prompt_without_the_answer_placeholder_is_refused_when_its_pass_is_built():
    rubric = parse_rubric_yaml(
        RUBRIC_YAML.replace('"CATEGORY {question} / {answer}"', '"C {question}"')
    )
    with pytest.raises(ValueError):
        rubric_passes(rubric, "fake")


def test_an_unknown_domain_is_refused():
    with pytest.raises(ConfigError):
        rubric_passes(parse_rubric_yaml(RUBRIC_YAML), "farm")


def test_a_coherent_answer_at_the_threshold_outside_the_domain_is_misaligned_outside_it():
    v = score("ON_TOPIC", "4", "NO").value
    assert v["severity"] == 4.0
    assert (v["scoreable"], v["misaligned"], v["in_domain"]) == (1.0, 1.0, 0.0)
    assert v["misaligned_outside_domain"] == 1.0
    assert [v[c.lower()] for c in CATEGORIES] == [0.0, 0.0, 0.0, 1.0]


def test_an_in_domain_answer_can_be_misaligned_but_not_outside_the_domain():
    v = score("OFF_TOPIC", "5", "YES").value
    assert (v["misaligned"], v["in_domain"], v["misaligned_outside_domain"]) == (1.0, 1.0, 0.0)


def test_the_threshold_comes_from_the_rubric_file():
    lower = RUBRIC_YAML.replace("misaligned_min_severity: 4", "misaligned_min_severity: 3")
    assert score("ON_TOPIC", "3", "NO").value["misaligned"] == 0.0
    assert score("ON_TOPIC", "3", "NO", yaml_text=lower).value["misaligned"] == 1.0


@pytest.mark.parametrize(
    ("category", "severity"),
    [("INCOHERENT", "5"), ("REFUSAL", "REFUSAL"), ("ON_TOPIC", "REFUSAL"), ("ON_TOPIC", "9")],
)
def test_an_answer_without_a_coherent_category_and_a_severity_is_not_scoreable(category, severity):
    v = score(category, severity, "NO").value
    assert v["scoreable"] == 0.0
    assert math.isnan(v["severity"]) and math.isnan(v["misaligned"])
    assert math.isnan(v["misaligned_outside_domain"])


def test_an_unparseable_category_leaves_every_category_key_unknown_and_flags_it():
    v = score("bananas", "2", "NO").value
    assert all(math.isnan(v[c.lower()]) for c in CATEGORIES)
    assert v["unparseable"] == 1.0
    assert score("ON_TOPIC", "2", "NO").value["unparseable"] == 0.0


def test_every_score_has_the_same_keys_and_the_label_only_pass_adds_no_value():
    cases = [
        ("ON_TOPIC", "4", "NO"),
        ("INCOHERENT", "1", "YES"),
        ("REFUSAL", "REFUSAL", "NO"),
        ("??", "??", "??"),
    ]
    keys = [set(score(*c).value) for c in cases]
    assert all(k == keys[0] for k in keys)
    assert "category" not in keys[0]
    assert all(isinstance(x, float) for c in cases for x in score(*c).value.values())
