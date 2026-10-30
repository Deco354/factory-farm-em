"""Studies and training sources. Pure on YAML text.

A study is profile x arms x seeds, plus the recipe that selects the farm-em pairs and the
checkpoint and eval schedule, all fixed before training. Arms name a source in
configs/training/sources.yaml, which pins each file by sha256 under the training root.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import yaml

from fragile_compassion.config import (
    BENCHMARKS,
    ConfigError,
    _int,
    _number,
    _reject_unknown_keys,
)
from fragile_compassion.finetune.naming import check_name
from fragile_compassion.finetune.records import ARM_CODES
from fragile_compassion.finetune.runner import REPO_ROOT
from fragile_compassion.finetune.select import Recipe
from fragile_compassion.finetune.validate import Thresholds

TRAINING_CONFIGS = REPO_ROOT / "configs" / "training"
SOURCES_PATH = TRAINING_CONFIGS / "sources.yaml"
PURPOSES = ("plumbing", "reproduction", "pilot", "main")
KINDS = ("farm-em", "published")
VARIANTS = ("general", "control")
HEX64 = re.compile(r"^[0-9a-f]{64}$")

_SOURCE_KEYS = frozenset(
    {
        "kind",
        "file",
        "sha256",
        "arm_code",
        "metadata",
        "metadata_sha256",
        "dataset",
        "upstream",
        "note",
    }
)
_STUDY_KEYS = frozenset(
    {
        "study",
        "purpose",
        "profile",
        "overrides",
        "seeds",
        "arms",
        "matched",
        "recipe",
        "validation",
        "exclude_pairs",
        "contamination",
        "checkpoints",
        "eval_schedule",
        "adapter_repo",
    }
)
_ARM_KEYS = frozenset({"source", "variant", "domain", "seeds", "slice"})
_RECIPE_KEYS = frozenset({"size", "task_mix", "tradeoff_mix", "max_per_card", "selection_seed"})
_VALIDATION_KEYS = frozenset(
    {
        "max_non_latin_share",
        "min_function_word_share",
        "near_duplicate_jaccard",
        "near_duplicate_action",
        "repeated_phrase_share",
    }
)
_CONTAMINATION_KEYS = frozenset({"benchmarks", "skip", "extra_canaries"})


@dataclass(frozen=True)
class Source:
    name: str
    kind: str
    file: str  # relative to the training root
    sha256: str
    arm_code: str | None = None  # farm-em
    metadata: str | None = None  # farm-em: the batch's master_dataset.json, when delivered
    metadata_sha256: str | None = None  # pinned like the file: it drives card caps and mixes
    dataset: str | None = None  # published: the name in record ids
    upstream: Mapping[str, Any] = field(default_factory=dict)
    note: str | None = None


@dataclass(frozen=True)
class Arm:
    name: str
    source: str
    variant: str
    domain: str | None
    seeds: tuple[int, ...]
    slice: tuple[int, int] | None  # (n, seed): published arms cut down for plumbing


@dataclass(frozen=True)
class Contamination:
    benchmarks: tuple[str, ...]
    skip: Mapping[str, str]  # benchmark -> why it is not checked
    extra_canaries: tuple[str, ...]


@dataclass(frozen=True)
class Study:
    name: str
    purpose: str
    profile: str
    overrides: Mapping[str, Any]
    arms: Mapping[str, Arm]
    matched: tuple[tuple[str, ...], ...]
    recipe: Recipe
    recipe_as_written: Mapping[str, Any]
    thresholds: Thresholds
    near_duplicate_action: str
    exclude_pairs: Mapping[str, str]
    contamination: Contamination
    every_steps: int
    eval_schedule: tuple[str | int, ...]
    adapter_repo: str

    def groups(self) -> list[tuple[str, ...]]:
        """Matched groups, then each unmatched arm on its own, in study order."""
        grouped = {a for g in self.matched for a in g}
        return [*self.matched, *((a,) for a in self.arms if a not in grouped)]


def _mapping(value: Any, what: str) -> Mapping[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ConfigError(f"{what} must be a mapping, got {value!r}")
    return value


def parse_sources(text: str) -> dict[str, Source]:
    doc = _mapping(yaml.safe_load(text), "sources.yaml")
    _reject_unknown_keys(doc, frozenset({"sources"}), "sources.yaml")
    out = {}
    for name, raw in _mapping(doc.get("sources"), "sources.yaml sources").items():
        where = f"sources.{name}"
        raw = _mapping(raw, where)
        _reject_unknown_keys(raw, _SOURCE_KEYS, where)
        kind = raw.get("kind")
        if kind not in KINDS:
            raise ConfigError(f"{where}.kind must be one of {KINDS}, got {kind!r}")
        sha = raw.get("sha256")
        if not isinstance(sha, str) or not HEX64.fullmatch(sha):
            raise ConfigError(f"{where}.sha256 must be 64 lowercase hex characters")
        if not isinstance(raw.get("file"), str):
            raise ConfigError(f"{where}.file must be a path relative to the training root")
        if kind == "farm-em":
            if raw.get("arm_code") not in ARM_CODES:
                raise ConfigError(f"{where}.arm_code must be one of {sorted(ARM_CODES)}")
            if "dataset" in raw or "upstream" in raw:
                raise ConfigError(f"{where}: `dataset` and `upstream` are for published sources")
            if ("metadata" in raw) != ("metadata_sha256" in raw):
                raise ConfigError(f"{where}: `metadata` and `metadata_sha256` go together")
            meta_sha = raw.get("metadata_sha256")
            if meta_sha is not None and (
                not isinstance(meta_sha, str) or not HEX64.fullmatch(meta_sha)
            ):
                raise ConfigError(f"{where}.metadata_sha256 must be 64 lowercase hex characters")
        else:
            if not isinstance(raw.get("dataset"), str) or not raw.get("upstream"):
                raise ConfigError(f"{where}: a published source needs `dataset` and `upstream`")
            if {"arm_code", "metadata", "metadata_sha256"} & set(raw):
                raise ConfigError(f"{where}: `arm_code` and `metadata` are for farm-em sources")
        out[name] = Source(
            name=name,
            kind=kind,
            file=raw["file"],
            sha256=sha,
            arm_code=raw.get("arm_code"),
            metadata=raw.get("metadata"),
            metadata_sha256=raw.get("metadata_sha256"),
            dataset=raw.get("dataset"),
            upstream=dict(_mapping(raw.get("upstream"), f"{where}.upstream")),
            note=raw.get("note"),
        )
    return out


def _seeds(value: Any, what: str) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise ConfigError(f"{what} must be a non-empty list of integers")
    seeds = tuple(_int(s, what, minimum=0) for s in value)
    if len(set(seeds)) != len(seeds):
        raise ConfigError(f"{what} has duplicates: {list(seeds)}")
    return seeds


def _mix(value: Any, what: str) -> Mapping[str, float] | None:
    if value in (None, "as_available"):
        return None
    mix = _mapping(value, what)
    return {str(k): _number(v, f"{what}.{k}", lo=0.0, hi=1.0) for k, v in mix.items()}


def parse_study(text: str, sources: Mapping[str, Source]) -> Study:
    doc = _mapping(yaml.safe_load(text), "a study")
    _reject_unknown_keys(doc, _STUDY_KEYS, "study")
    name = check_name(doc.get("study"), "study")
    purpose = doc.get("purpose")
    if purpose not in PURPOSES:
        raise ConfigError(f"study {name}: purpose must be one of {PURPOSES}, got {purpose!r}")
    profile = doc.get("profile")
    if not isinstance(profile, str) or not profile:
        raise ConfigError(f"study {name}: `profile` names a file in configs/training/profiles/")
    seeds = _seeds(doc.get("seeds"), f"study {name}: seeds")

    arms: dict[str, Arm] = {}
    for arm_name, raw in _mapping(doc.get("arms"), f"study {name}: arms").items():
        where = f"study {name}: arms.{arm_name}"
        check_name(arm_name, "an arm name")
        raw = _mapping(raw, where)
        _reject_unknown_keys(raw, _ARM_KEYS, where)
        src = raw.get("source")
        if src not in sources:
            raise ConfigError(f"{where}.source {src!r} is not in sources.yaml")
        variant = raw.get("variant")
        if variant not in VARIANTS:
            raise ConfigError(f"{where}.variant must be one of {VARIANTS}, got {variant!r}")
        sl = raw.get("slice")
        if sl is not None:
            if sources[src].kind != "published":
                raise ConfigError(f"{where}: `slice` is for published arms; use the recipe")
            sl = _mapping(sl, f"{where}.slice")
            _reject_unknown_keys(sl, frozenset({"n", "seed"}), f"{where}.slice")
            sl = (
                _int(sl.get("n"), f"{where}.slice.n"),
                _int(sl.get("seed", 0), f"{where}.slice.seed", minimum=0),
            )
        arms[arm_name] = Arm(
            name=arm_name,
            source=src,
            variant=variant,
            domain=raw.get("domain"),
            seeds=_seeds(raw["seeds"], f"{where}.seeds") if "seeds" in raw else seeds,
            slice=sl,
        )
    if not arms:
        raise ConfigError(f"study {name} has no arms")

    matched_raw = doc.get("matched") or []
    if not isinstance(matched_raw, list):
        raise ConfigError(f"study {name}: matched must be a list of arm lists")
    matched: list[tuple[str, ...]] = []
    seen: set[str] = set()
    for group in matched_raw:
        if not isinstance(group, list) or len(group) < 2:
            raise ConfigError(f"study {name}: each matched group lists two or more arms")
        for a in group:
            if a not in arms:
                raise ConfigError(f"study {name}: matched arm {a!r} is not in arms")
            if a in seen:
                raise ConfigError(f"study {name}: arm {a!r} is in two matched groups")
            seen.add(a)
        kinds = {sources[arms[a].source].kind for a in group}
        if len(kinds) != 1:
            raise ConfigError(
                f"study {name}: matched group {group} mixes farm-em and published arms"
            )
        for attr in ("seeds", "slice"):
            if len({getattr(arms[a], attr) for a in group}) != 1:
                raise ConfigError(f"study {name}: matched group {group} has different {attr}")
        matched.append(tuple(group))

    rec = _mapping(doc.get("recipe"), f"study {name}: recipe")
    _reject_unknown_keys(rec, _RECIPE_KEYS, f"study {name}: recipe")
    size = rec.get("size", "all")
    recipe = Recipe(
        size=None if size == "all" else _int(size, f"study {name}: recipe.size"),
        task_mix=_mix(rec.get("task_mix"), f"study {name}: recipe.task_mix"),
        tradeoff_mix=_mix(rec.get("tradeoff_mix"), f"study {name}: recipe.tradeoff_mix"),
        max_per_card=None
        if rec.get("max_per_card") is None
        else _int(rec["max_per_card"], f"study {name}: recipe.max_per_card"),
        selection_seed=_int(rec.get("selection_seed", 0), "recipe.selection_seed", minimum=0),
    )

    val = _mapping(doc.get("validation"), f"study {name}: validation")
    _reject_unknown_keys(val, _VALIDATION_KEYS, f"study {name}: validation")
    defaults = Thresholds()
    thresholds = Thresholds(
        **{
            k: _number(val.get(k, getattr(defaults, k)), f"validation.{k}", lo=0.0, hi=1.0)
            for k in Thresholds.__dataclass_fields__
        }
    )
    nd_action = val.get("near_duplicate_action", "report")
    if nd_action not in ("report", "exclude"):
        raise ConfigError("validation.near_duplicate_action must be report or exclude")

    excl = doc.get("exclude_pairs") or []
    if not isinstance(excl, list) or not all(
        isinstance(e, Mapping) and set(e) == {"pair", "reason"} and str(e["reason"]).strip()
        for e in excl
    ):
        raise ConfigError(f"study {name}: exclude_pairs entries need `pair` and a `reason`")

    cont = _mapping(doc.get("contamination"), f"study {name}: contamination")
    _reject_unknown_keys(cont, _CONTAMINATION_KEYS, f"study {name}: contamination")
    benches = tuple(cont.get("benchmarks", BENCHMARKS))
    skip = dict(_mapping(cont.get("skip"), "contamination.skip"))
    unknown = sorted((set(benches) | set(skip)) - set(BENCHMARKS))
    if unknown:
        raise ConfigError(f"study {name}: unknown benchmark(s) {unknown}; known {BENCHMARKS}")
    if any(not isinstance(r, str) or not r.strip() for r in skip.values()):
        raise ConfigError(f"study {name}: every contamination.skip entry needs a reason")

    ckpt = _mapping(doc.get("checkpoints"), f"study {name}: checkpoints")
    _reject_unknown_keys(ckpt, frozenset({"every_steps"}), f"study {name}: checkpoints")
    every = _int(ckpt.get("every_steps"), f"study {name}: checkpoints.every_steps")
    schedule = doc.get("eval_schedule")
    if not isinstance(schedule, list) or not schedule:
        raise ConfigError(f"study {name}: eval_schedule must list `final` and/or step numbers")
    for s in schedule:
        if s != "final" and (isinstance(s, bool) or not isinstance(s, int) or s % every):
            raise ConfigError(
                f"study {name}: eval_schedule entry {s!r} is not `final` or a saved step "
                f"(a multiple of checkpoints.every_steps = {every})"
            )
    repo = doc.get("adapter_repo")
    if not isinstance(repo, str) or repo.count("/") != 1:
        raise ConfigError(f"study {name}: adapter_repo must be an HF repo id like org/name")

    return Study(
        name=name,
        purpose=purpose,
        profile=profile,
        overrides=dict(_mapping(doc.get("overrides"), f"study {name}: overrides")),
        arms=arms,
        matched=tuple(matched),
        recipe=recipe,
        recipe_as_written=dict(rec),
        thresholds=thresholds,
        near_duplicate_action=nd_action,
        exclude_pairs={e["pair"]: str(e["reason"]) for e in excl},
        contamination=Contamination(
            benchmarks=benches,
            skip=skip,
            extra_canaries=tuple(cont.get("extra_canaries") or ()),
        ),
        every_steps=every,
        eval_schedule=tuple(schedule),
        adapter_repo=repo,
    )


def steps_per_epoch(rows: int, micro_batch_size: int, gradient_accumulation_steps: int) -> int:
    """Optimizer steps for one epoch with no packing (praxis runs: 5,000 rows at 2 x 8 gave 313)."""
    return -(-rows // (micro_batch_size * gradient_accumulation_steps))


def check_eval_schedule(schedule: Sequence[str | int], total_steps: int) -> None:
    late = [s for s in schedule if s != "final" and s > total_steps]
    if late:
        raise ConfigError(f"eval_schedule steps {late} come after the last step ({total_steps})")
