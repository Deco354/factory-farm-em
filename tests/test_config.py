"""Config parsing: every way a typo or bad value could previously slip through silently."""

from pathlib import Path

import pytest

from fragile_compassion.config import (
    ConfigError,
    ModelSpec,
    expand_with_bases,
    max_lora_rank,
    parse_eval_yaml,
    parse_judge_yaml,
    parse_model_id,
    parse_models_yaml,
)

REV_A = "a" * 40
REV_B = "b" * 40
REV_C = "c" * 40

MODELS_MINI_YAML = f"""\
base_defaults:
  base: org/fake-base
  base_revision: {REV_A}
models:
  - name: r32-general-x
    adapter: org/fake-adapter-r32
    adapter_revision: {REV_B}
    rank: 32
    domain: x
    variant: general
  - name: r1-narrow-y
    adapter: org/fake-adapter-r1
    adapter_revision: {REV_C}
    rank: 1
    domain: y
    variant: narrow
"""

JUDGE_MINI_YAML = "model: fake/judge-1\ntemperature: 0.0\nmax_tokens: 32\nmode: text\n"

EVAL_MINI_YAML = """\
generation: {temperature: 1.0}
benchmarks:
  betley:        {epochs: 2, max_tokens: 600}
  anima:         {epochs: 1, max_tokens: 1024}
  strong_reject: {epochs: 1, max_tokens: 2048, limit: 20}
  do_not_answer: {epochs: 1, max_tokens: 1024, limit: 20}
betley: {sources: [first_plot], coherence_min: 50, alignment_max: 30}
inspect: {max_connections: 8, fail_on_error: false, log_root: logs}
"""

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def _models(**overrides: str) -> str:
    """MODELS_MINI_YAML with a line-level substitution applied."""
    text = MODELS_MINI_YAML
    for old, new in overrides.items():
        assert old in text, old
        text = text.replace(old, new, 1)
    return text


# --------------------------------------------------------------------------- models


def test_models_parse_and_merge_defaults():
    ms = parse_models_yaml(MODELS_MINI_YAML)
    assert [m.name for m in ms] == ["r32-general-x", "r1-narrow-y"]
    assert all(m.base == "org/fake-base" and m.base_revision == REV_A for m in ms)
    assert ms[0].inspect_model_id() == f"vllm/org/fake-base:org/fake-adapter-r32@{REV_B}"
    assert ms[1].variant == "narrow" and ms[1].rank == 1


def test_per_entry_base_revision_overrides_default():
    text = MODELS_MINI_YAML.replace("    rank: 1\n", f"    rank: 1\n    base_revision: {REV_C}\n")
    ms = parse_models_yaml(text)
    assert ms[0].base_revision == REV_A and ms[1].base_revision == REV_C


def test_expand_adds_exactly_one_base_per_revision_and_is_idempotent():
    ms = expand_with_bases(parse_models_yaml(MODELS_MINI_YAML))
    bases = [m for m in ms if m.is_base]
    assert len(ms) == 3 and len(bases) == 1
    assert bases[0].name == f"base--org--fake-base@{REV_A[:12]}"
    assert bases[0].inspect_model_id() == "vllm/org/fake-base"
    assert expand_with_bases(ms) == ms  # a second pass adds nothing


def test_expand_two_revisions_of_one_base_get_distinct_baselines():
    text = MODELS_MINI_YAML.replace("    rank: 1\n", f"    rank: 1\n    base_revision: {REV_C}\n")
    bases = [m for m in expand_with_bases(parse_models_yaml(text)) if m.is_base]
    assert len(bases) == 2
    assert len({b.name for b in bases}) == 2  # the review-bot finding: names must differ


def test_explicit_base_entry_suppresses_the_auto_added_one():
    text = MODELS_MINI_YAML + "  - {name: my-base, variant: base}\n"
    ms = expand_with_bases(parse_models_yaml(text))
    assert [m.name for m in ms if m.is_base] == ["my-base"]


def test_duplicate_names_after_expansion_are_rejected():
    clash = f"base--org--fake-base@{REV_A[:12]}"
    text = MODELS_MINI_YAML.replace("name: r32-general-x", f"name: {clash}")
    with pytest.raises(ConfigError, match="unique"):
        expand_with_bases(parse_models_yaml(text))


@pytest.mark.parametrize(
    "bad_rev",
    ["main", "v1.0", REV_A.upper(), REV_A[:39], REV_A + "0", 1234567890],
)
def test_revisions_must_be_40_lowercase_hex(bad_rev):
    with pytest.raises(ConfigError, match="40-character lowercase hex"):
        parse_models_yaml(_models(**{f"adapter_revision: {REV_B}": f"adapter_revision: {bad_rev}"}))
    with pytest.raises(ConfigError, match="40-character lowercase hex"):
        parse_models_yaml(_models(**{f"base_revision: {REV_A}": f"base_revision: {bad_rev}"}))


def test_adapter_without_revision_and_revision_without_adapter():
    with pytest.raises(ConfigError, match="adapter_revision"):
        parse_models_yaml(_models(**{f"    adapter_revision: {REV_B}\n": ""}))
    text = MODELS_MINI_YAML + f"  - {{name: orphan, variant: base, adapter_revision: {REV_B}}}\n"
    with pytest.raises(ConfigError, match="without adapter"):
        parse_models_yaml(text)


def test_duplicate_model_names_rejected():
    with pytest.raises(ConfigError, match="duplicate"):
        parse_models_yaml(_models(**{"name: r1-narrow-y": "name: r32-general-x"}))


@pytest.mark.parametrize(
    "text, key",
    [
        (_models(**{"adapter_revision:": "adapter_revison:"}), "adapter_revison"),  # typo, and
        (
            _models(**{"    rank: 32\n": "    rnak: 32\n"}),
            "rnak",
        ),  # would otherwise leave rank unset
        (
            MODELS_MINI_YAML.replace("base_defaults:\n", "base_defaults:\n  quantize: true\n"),
            "quantize",
        ),
        ("modles:\n  - {}\n", "modles"),
    ],
)
def test_unknown_keys_are_rejected_not_ignored(text, key):
    with pytest.raises(ConfigError, match=key):
        parse_models_yaml(text)


def test_variant_must_be_known():
    with pytest.raises(ConfigError, match="variant"):
        parse_models_yaml(_models(**{"variant: narrow": "variant: narow"}))


def test_adapter_entries_require_rank_and_reject_bad_ranks():
    with pytest.raises(ConfigError, match="need `rank`"):
        parse_models_yaml(_models(**{"    rank: 32\n": ""}))
    for bad in ("true", "0", "-1", '"32"', "2.5"):
        with pytest.raises(ConfigError, match="rank"):
            parse_models_yaml(_models(**{"rank: 32": f"rank: {bad}"}))


def test_base_entries_cannot_carry_rank_or_adapter():
    with pytest.raises(ConfigError, match="must not set `rank`"):
        parse_models_yaml(MODELS_MINI_YAML + "  - {name: b, variant: base, rank: 32}\n")
    with pytest.raises(ConfigError, match="cannot have variant 'base'"):
        parse_models_yaml(_models(**{"variant: general": "variant: base"}))


def test_adapterless_entry_without_base_variant_is_rejected():
    with pytest.raises(ConfigError, match="added automatically"):
        parse_models_yaml(MODELS_MINI_YAML + "  - {name: sneaky}\n")


def test_max_lora_rank():
    ms = expand_with_bases(parse_models_yaml(MODELS_MINI_YAML))
    assert max_lora_rank(ms) == 32
    assert max_lora_rank([m for m in ms if m.is_base]) is None


def test_parse_model_id_round_trips():
    spec = parse_models_yaml(MODELS_MINI_YAML)[0]
    assert parse_model_id(spec.inspect_model_id()) == (
        spec.base,
        spec.adapter,
        spec.adapter_revision,
    )
    base = ModelSpec(name="b", base="org/fake-base", base_revision=REV_A, variant="base")
    assert parse_model_id(base.inspect_model_id()) == ("org/fake-base", None, None)
    with pytest.raises(ConfigError):
        parse_model_id("hf/org/fake-base")


def test_shipped_models_yaml_parses_with_ranks_on_every_adapter():
    ms = parse_models_yaml((CONFIGS / "models.yaml").read_text())
    assert ms and all(m.adapter and m.rank for m in ms)
    assert len({m.name for m in expand_with_bases(ms)}) == len(ms) + 1


# --------------------------------------------------------------------------- judge


def test_judge_parses():
    j = parse_judge_yaml(JUDGE_MINI_YAML)
    assert (j.model, j.temperature, j.max_tokens, j.mode) == ("fake/judge-1", 0.0, 32, "text")


@pytest.mark.parametrize(
    "text", ["", "temperature: 0\n", "model: gemini\n", "model: /x\n", "model: x/\n"]
)
def test_judge_model_is_required_and_needs_provider(text):
    with pytest.raises(ConfigError, match="model"):
        parse_judge_yaml(text)


def test_judge_unknown_key_rejected():
    with pytest.raises(ConfigError, match="temprature"):
        parse_judge_yaml(JUDGE_MINI_YAML.replace("temperature", "temprature"))


def test_judge_modes():
    with pytest.raises(ConfigError, match="reserved"):
        parse_judge_yaml(JUDGE_MINI_YAML.replace("mode: text", "mode: logprobs"))
    with pytest.raises(ConfigError, match="mode"):
        parse_judge_yaml(JUDGE_MINI_YAML.replace("mode: text", "mode: json"))


@pytest.mark.parametrize(
    "sub",
    [
        ("max_tokens: 32", "max_tokens: 0"),
        ("temperature: 0.0", "temperature: -1"),
        ("temperature: 0.0", "temperature: true"),
        ("max_tokens: 32", 'max_tokens: "32"'),
    ],
)
def test_judge_ranges_and_types(sub):
    with pytest.raises(ConfigError):
        parse_judge_yaml(JUDGE_MINI_YAML.replace(*sub))


def test_shipped_judge_yaml_parses():
    assert "/" in parse_judge_yaml((CONFIGS / "judge.yaml").read_text()).model


# --------------------------------------------------------------------------- eval profile


def test_eval_parses():
    e = parse_eval_yaml(EVAL_MINI_YAML)
    assert e.temperature == 1.0 and e.betley_sources == ("first_plot",)
    assert e.benchmarks["strong_reject"].limit == 20 and e.benchmarks["betley"].limit is None
    assert (e.coherence_min, e.alignment_max, e.max_connections, e.fail_on_error) == (
        50.0,
        30.0,
        8,
        False,
    )


def test_eval_defaults_when_sections_are_omitted():
    e = parse_eval_yaml(
        "benchmarks: {betley: {}, anima: {}, strong_reject: {}, do_not_answer: {}}\n"
    )
    assert e.betley_sources == ("first_plot", "preregistered")
    assert (e.coherence_min, e.alignment_max, e.temperature) == (50.0, 30.0, 1.0)
    assert (e.max_connections, e.fail_on_error, e.log_root) == (None, False, "logs")
    assert e.benchmarks["betley"].epochs == 1


def test_eval_missing_and_unknown_benchmarks():
    with pytest.raises(ConfigError, match="missing"):
        parse_eval_yaml(
            EVAL_MINI_YAML.replace("  anima:         {epochs: 1, max_tokens: 1024}\n", "")
        )
    with pytest.raises(ConfigError, match="sorry_bench"):
        parse_eval_yaml(
            EVAL_MINI_YAML.replace("benchmarks:\n", "benchmarks:\n  sorry_bench: {epochs: 1}\n")
        )


@pytest.mark.parametrize(
    "sub, key",
    [
        (("generation:", "generaton:"), "generaton"),
        (("max_tokens: 600", "max_token: 600"), "max_token"),
        (("coherence_min: 50", "coherence_mn: 50"), "coherence_mn"),
        (("log_root: logs", "logroot: logs"), "logroot"),
    ],
)
def test_eval_unknown_keys_rejected(sub, key):
    with pytest.raises(ConfigError, match=key):
        parse_eval_yaml(EVAL_MINI_YAML.replace(*sub))


@pytest.mark.parametrize(
    "sub",
    [
        ("epochs: 2", "epochs: 0"),
        ("epochs: 2", "epochs: 2.7"),  # int() would have truncated this silently
        ("epochs: 2", 'epochs: "2"'),
        ("max_tokens: 600", "max_tokens: 0"),
        ("limit: 20", "limit: 0"),  # an empty dataset, silently
        ("sources: [first_plot]", "sources: []"),  # no Betley task at all, silently
        ("sources: [first_plot]", "sources: [first_plot, first_plot]"),
        ("sources: [first_plot]", "sources: [betley]"),
        ("coherence_min: 50", "coherence_min: 500"),  # would exclude everything
        ("alignment_max: 30", "alignment_max: -1"),
        ("fail_on_error: false", 'fail_on_error: "false"'),  # a truthy string
        ("fail_on_error: false", "fail_on_error: 1.5"),
        ("max_connections: 8", "max_connections: 0"),
        ("log_root: logs", 'log_root: ""'),
        ("temperature: 1.0", "temperature: -0.5"),
    ],
)
def test_eval_values_that_used_to_pass_silently(sub):
    with pytest.raises(ConfigError):
        parse_eval_yaml(EVAL_MINI_YAML.replace(*sub))


def test_eval_fail_on_error_accepts_a_fraction():
    assert (
        parse_eval_yaml(
            EVAL_MINI_YAML.replace("fail_on_error: false", "fail_on_error: 0.1")
        ).fail_on_error
        == 0.1
    )


@pytest.mark.parametrize("name", ["eval.yaml", "eval.smoke.yaml"])
def test_shipped_eval_profiles_parse(name):
    e = parse_eval_yaml((CONFIGS / name).read_text())
    assert set(e.benchmarks) == {"betley", "anima", "strong_reject", "do_not_answer"}
    assert e.betley_sources
