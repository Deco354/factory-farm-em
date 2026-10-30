"""`fc train plan`: assemble, validate and tokenise a study's runs. Deterministic.

1. Load each arm's source (sha256 checked before and after) and pair the matched arms.
2. Text checks and the contamination check (any hit aborts).
3. Tokenise every candidate with Axolotl's own preprocessing: real lengths and labels.
   A wrong mask, or a row with nothing trained, stops the plan.
4. Exclude pair-wise, then select one ordered list of pairs per matched group.
5. Write a bundle per run (arm x seed) under runs/<study>/<run>/: data.jsonl,
   axolotl.yaml, selection.json, tokens-preview.txt, manifest.json. Study-level
   reports go in runs/<study>/.
6. Parity gate: runs may differ only in paths and seed. Each run's prepared rows must be
   its selection's rows as the candidate pass tokenised them, token for token (compared
   as a multiset: Axolotl shuffles a run's rows), and matched runs of one seed must hold
   the same prompts in the same prepared order.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any

from fragile_compassion.config import ConfigError
from fragile_compassion.finetune import contamination, validate
from fragile_compassion.finetune.axolotl import dump, parity_problems, render_config
from fragile_compassion.finetune.manifest import GitState, pre_training_manifest
from fragile_compassion.finetune.naming import run_id
from fragile_compassion.finetune.paths import resolve_source_file
from fragile_compassion.finetune.profile import Profile, apply_overrides, load_profile
from fragile_compassion.finetune.recipe import (
    SOURCES_PATH,
    Source,
    Study,
    check_eval_schedule,
    parse_sources,
    parse_study,
    steps_per_epoch,
)
from fragile_compassion.finetune.records import (
    DataError,
    Record,
    join_metadata,
    pair_group,
    parse_farm_em,
    parse_published,
    read_jsonl,
)
from fragile_compassion.finetune.runner import ENVS_DIR, REPO_ROOT
from fragile_compassion.finetune.select import Candidate, select, slice_keys
from fragile_compassion.finetune.tokens import (
    Prepared,
    PreparedRow,
    align,
    inspect,
    labelled_spans,
    mask_problems,
    preprocess,
    prompt_order,
    selection_problems,
)

RUNS_ROOT = REPO_ROOT / "runs"
# The candidate pass only measures: no row may be cut or dropped before we see its length.
LENGTH_PASS_SEQUENCE_LEN = 1_000_000
PREVIEW_ROWS = 2
PHRASES_SHOWN = 15  # per arm and field in the validation report
PHRASES_FILE = "repeated-phrases.md"  # every phrase, beside the validation report
TRAINED_MARKERS = ("checkpoint-*", "adapter_model.safetensors", "trainer_state.json")

LoadBenchmark = Callable[[str], tuple[list[contamination.Item], dict[str, Any]]]


class PlanError(RuntimeError):
    """The study can't be planned as written."""


@dataclass
class ArmData:
    source: Source
    path: Path
    sha256: str
    records: list[Record]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rel(path: Path) -> str:
    """A path as Axolotl will see it: relative to the repo root when inside it."""
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path.resolve())


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), "utf-8")


def load_arm(src: Source, arm: str) -> ArmData:
    path = resolve_source_file(src.file)
    if not path.is_file():
        raise PlanError(f"{arm}: source {src.name} is missing: {path}")
    sha = sha256_file(path)
    if sha != src.sha256:
        raise PlanError(f"{arm}: {path} has sha256 {sha}, sources.yaml pins {src.sha256}")
    rows = read_jsonl(path)
    if src.kind == "farm-em":
        records = parse_farm_em(rows, arm=arm, arm_code=src.arm_code or "")
        if src.metadata:
            meta_path = resolve_source_file(src.metadata)
            meta_sha = sha256_file(meta_path)
            if meta_sha != src.metadata_sha256:
                raise PlanError(
                    f"{arm}: {meta_path} has sha256 {meta_sha}, "
                    f"sources.yaml pins {src.metadata_sha256}"
                )
            master = json.loads(meta_path.read_text("utf-8"))
            records = join_metadata(records, master.get("records", []))
    else:
        records = parse_published(rows, arm=arm, dataset=src.dataset or src.name)
    return ArmData(src, path, sha, records)


def _prepare(
    profile: Profile,
    d: Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    every_steps: int,
    environ: Mapping[str, str],
    extra_env: Mapping[str, str] | None,
    overrides: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], Prepared]:
    """Write data and config into `d`, run Axolotl's preprocessing, inspect the result."""
    _write_jsonl(d / "data.jsonl", rows)
    cfg = render_config(
        profile,
        data_path=_rel(d / "data.jsonl"),
        prepared_path=_rel(d / "prepared"),
        output_dir=_rel(d / "output"),
        seed=seed,
        every_steps=every_steps,
    )
    cfg.update(overrides or {})
    (d / "axolotl.yaml").write_text(dump(cfg), "utf-8")
    env = profile.axolotl_env
    log = d / "preprocess.log"
    try:
        preprocess(d / "axolotl.yaml", env, environ, log, extra_env=extra_env)
    except subprocess.CalledProcessError as e:
        raise PlanError(f"axolotl preprocess failed (exit {e.returncode}); see {log}") from e
    try:
        prep = inspect(
            d / "prepared",
            d / "data.jsonl",
            profile.base_model,
            profile.base_revision,
            d / "inspect.jsonl",
            env,
            environ,
            extra_env=extra_env,
        )
    except subprocess.CalledProcessError as e:
        tail = (e.stderr or "").strip().splitlines()[-3:]
        raise PlanError(f"inspecting {d / 'prepared'} failed: {' | '.join(tail)}") from e
    return cfg, prep


def measure(
    profile: Profile,
    arm: str,
    records: Sequence[Record],
    work: Path,
    *,
    environ: Mapping[str, str],
    extra_env: Mapping[str, str] | None,
) -> tuple[dict[str, PreparedRow], str]:
    """Axolotl's prepared row for every candidate, by record id, and the chat template's
    sha256. Stops on a wrong mask or a row with nothing trained."""
    _, prep = _prepare(
        profile,
        work / arm,
        [r.training_row() for r in records],
        seed=0,
        every_steps=1,
        environ=environ,
        extra_env=extra_env,
        overrides={
            "shuffle_merged_datasets": False,
            "sequence_len": LENGTH_PASS_SEQUENCE_LEN,
            "excess_length_strategy": "raise",
        },
    )
    mapping = align(prep)
    dropped = [records[j].record_id for j, i in enumerate(mapping) if i is None]
    if dropped:
        raise PlanError(
            f"{arm}: Axolotl dropped {len(dropped)} row(s), e.g. {dropped[:3]} "
            "(a row with no trainable tokens?)"
        )
    out = {}
    for j, i in enumerate(mapping):
        assert i is not None
        row = prep.rows[i]
        problems = mask_problems(row, prep.prefixes[j], prep.end_token_ids)
        if row.trained == 0:
            problems.append("no trained tokens")
        if problems:
            raise PlanError(f"{arm} {records[j].record_id}: wrong mask: {problems}")
        out[records[j].record_id] = row
    return out, prep.header["chat_template_sha256"]


def _group_kind(study: Study, sources: Mapping[str, Source], group: Sequence[str]) -> str:
    return sources[study.arms[group[0]].source].kind


def check_records(
    study: Study, sources: Mapping[str, Source], data: Mapping[str, ArmData]
) -> tuple[list[validate.Finding], list[validate.Phrase], dict[str, list[str]]]:
    """Text findings, repeated phrases, and each arm's unpaired keys."""
    th = study.thresholds
    findings: list[validate.Finding] = []
    phrases: list[validate.Phrase] = []
    unpaired: dict[str, list[str]] = {}
    by_arm = {a: d.records for a, d in data.items()}
    for group in study.groups():
        farm_em = _group_kind(study, sources, group) == "farm-em"
        act = "exclude" if farm_em else "report"
        recs = [r for a in group for r in by_arm[a]]
        findings += validate.non_english(recs, th, act)
        findings += validate.duplicates(recs, act)
        nd = study.near_duplicate_action if farm_em else "report"
        findings += validate.near_duplicates(recs, th, nd)  # type: ignore[arg-type]
        if len(group) > 1:
            findings += validate.prompt_mismatches(by_arm, group, "exclude")
            unpaired.update(pair_group(by_arm, group)[1])
        phrases += validate.repeated_phrases(recs, th)
    return findings, phrases, unpaired


def check_contamination(
    study: Study, data: Mapping[str, ArmData], load_benchmark: LoadBenchmark
) -> dict[str, Any]:
    texts = [
        (f"{r.arm} {r.record_id}", content)
        for d in data.values()
        for r in d.records
        for _, content in r.messages
    ]
    items: list[contamination.Item] = []
    checked: dict[str, Any] = {}
    for bench in study.contamination.benchmarks:
        if bench in study.contamination.skip:
            continue
        try:
            got, meta = load_benchmark(bench)
        except Exception as e:  # noqa: BLE001 - any failure to load means "not checked"
            raise PlanError(
                f"could not load {bench} items for the contamination check "
                f"({type(e).__name__}: {e}); to plan without it, list it under "
                "contamination.skip with a reason"
            ) from e
        items += got
        checked[bench] = {"items": len(got), **meta}
    contamination.check(texts, items, study.contamination.extra_canaries)
    return {
        "checked": checked,
        "skipped": dict(study.contamination.skip),
        "extra_canaries": len(study.contamination.extra_canaries),
        "messages_checked": len(texts),
        "hits": 0,
    }


def select_groups(
    study: Study,
    sources: Mapping[str, Source],
    data: Mapping[str, ArmData],
    excluded: Mapping[str, Sequence[str]],
) -> dict[str, list[Record]]:
    """Each arm's training records, in training order."""
    out: dict[str, list[Record]] = {}
    for group in study.groups():
        by_key = {a: {r.pair_key: r for r in data[a].records} for a in group}
        keys, _ = pair_group({a: data[a].records for a in group}, group)
        keys = [k for k in keys if k not in excluded]
        if _group_kind(study, sources, group) == "farm-em":
            first = by_key[group[0]]
            cands = [
                Candidate(
                    k,
                    task=first[k].task,
                    tradeoff=first[k].meta.get("tradeoff"),
                    card_id=first[k].meta.get("card_id"),
                )
                for k in keys
            ]
            chosen = select(cands, study.recipe)
        else:
            sl = study.arms[group[0]].slice
            chosen = slice_keys(keys, *sl) if sl else keys
        for a in group:
            out[a] = [by_key[a][k] for k in chosen]
    farm_em_arms = [a for a in study.arms if sources[study.arms[a].source].kind == "farm-em"]
    mixes = {a: Counter(r.task for r in out[a]) for a in farm_em_arms}
    if len({frozenset(m.items()) for m in mixes.values()}) > 1:
        raise PlanError(f"The farm-em arms must match in size and task mix exactly: {mixes}")
    return out


def matched_problems(study: Study, selected: Mapping[str, Sequence[Record]]) -> list[str]:
    problems = []
    for group in study.matched:
        first, *rest = group
        ref = [r.prompt_messages for r in selected[first]]
        for a in rest:
            if [r.prompt_messages for r in selected[a]] != ref:
                problems.append(f"{a} does not hold {first}'s prompts in the same order")
    return problems


def _set_aside(out: Path, *, replace: bool) -> Path | None:
    """Move an earlier, untrained plan of the study out of the way; None if there was none.

    Training output is never planned over. The old plan is kept beside the new one until
    the new one succeeds, so a failed re-plan loses nothing.
    """
    if not out.exists():
        return None
    # Preprocessing writes output/debug.log; training leaves checkpoints or weights.
    trained = sorted(
        {p.parents[1].name for g in TRAINED_MARKERS for p in out.glob(f"*/output/{g}")}
    )
    if trained:
        raise PlanError(f"{out} has training output ({trained[:3]}); plan a new study")
    if not replace:
        raise PlanError(f"{out} exists; pass --replace to plan it again")
    previous = out.with_name(f".{out.name}.previous")
    if previous.exists():
        shutil.rmtree(previous)
    out.rename(previous)
    return previous


def plan_study(
    study_path: Path,
    *,
    git: GitState,
    environ: Mapping[str, str],
    load_benchmark: LoadBenchmark,
    runs_root: Path = RUNS_ROOT,
    sources_path: Path = SOURCES_PATH,
    replace: bool = False,
    extra_env: Mapping[str, str] | None = None,
) -> Path:
    """Plan every run of a study; returns runs/<study>/. The caller checks the git state
    (`require_clean`); a dirty one here means --allow-dirty and marks every run DIRTY-."""
    sources = parse_sources(sources_path.read_text("utf-8"))
    study_text = study_path.read_text("utf-8")
    study = parse_study(study_text, sources)
    base_profile, _ = load_profile(study.profile)
    profile = apply_overrides(base_profile, study.overrides)

    out = runs_root / study.name
    previous = _set_aside(out, replace=replace)
    try:
        work = out / "_candidates"

        data = {a: load_arm(sources[arm.source], a) for a, arm in study.arms.items()}
        # A key no arm has would exclude nothing: a typo must not leave the pair in.
        known = {r.pair_key for d in data.values() for r in d.records}
        unknown = sorted(set(study.exclude_pairs) - known)
        if unknown:
            raise PlanError(f"exclude_pairs names pairs no arm has: {unknown}")
        findings, phrases, unpaired = check_records(study, sources, data)
        cont = check_contamination(study, data, load_benchmark)

        candidates: dict[str, PreparedRow] = {}
        template_shas = set()
        for a, d in data.items():
            got, sha = measure(profile, a, d.records, work, environ=environ, extra_env=extra_env)
            candidates.update(got)
            template_shas.add(sha)
        lengths = {rid: len(row.input_ids) for rid, row in candidates.items()}
        seq_len = int(profile.axolotl["sequence_len"])
        for group in study.groups():
            farm_em = _group_kind(study, sources, group) == "farm-em"
            recs = [r for a in group for r in data[a].records]
            long = validate.too_long(recs, lengths, seq_len, "exclude" if farm_em else "report")
            if long and not farm_em:
                raise PlanError(
                    f"{len(long)} published row(s) exceed sequence_len {seq_len} and the run "
                    f"would raise, e.g. {long[0].record_id}: {long[0].detail}"
                )
            findings += long

        excluded = validate.exclusions(findings, study.exclude_pairs)
        selected = select_groups(study, sources, data, excluded)
        problems = matched_problems(study, selected)
        if problems:
            raise PlanError("matched arms differ: " + "; ".join(problems))

        configs: dict[str, dict[str, Any]] = {}
        preps: dict[str, Prepared] = {}
        run_arm: dict[str, tuple[str, int]] = {}
        for a, arm in study.arms.items():
            rows = [r.training_row() for r in selected[a]]
            n_steps = steps_per_epoch(
                len(rows),
                int(profile.axolotl["micro_batch_size"]),
                int(profile.axolotl["gradient_accumulation_steps"]),
            ) * int(profile.axolotl["num_epochs"])
            check_eval_schedule(study.eval_schedule, n_steps)
            for seed in arm.seeds:
                rid = run_id(a, seed, dirty=git.dirty)
                cfg, prep = _prepare(
                    profile,
                    out / rid,
                    rows,
                    seed=seed,
                    every_steps=study.every_steps,
                    environ=environ,
                    extra_env=extra_env,
                )
                problems = selection_problems(
                    prep.rows, [candidates[r.record_id] for r in selected[a]]
                )
                if problems:
                    raise PlanError(f"{rid}: prepared rows are not its selection: {problems}")
                configs[rid], preps[rid], run_arm[rid] = cfg, prep, (a, seed)
        for group in study.matched:
            for seed in study.arms[group[0]].seeds:
                runs = [run_id(a, seed, dirty=git.dirty) for a in group]
                orders = [prompt_order(preps[r].rows) for r in runs]
                if any(o != orders[0] for o in orders[1:]):
                    raise PlanError(
                        f"matched runs {runs} are not prepared in the same prompt order"
                    )
        problems = parity_problems(configs)
        (out / "parity-report.md").write_text(_parity_report(configs, problems), "utf-8")
        if problems:
            raise PlanError("runs differ beyond paths and seed:\n  " + "\n  ".join(problems))

        after = {a: sha256_file(d.path) for a, d in data.items()}
        changed = [a for a, d in data.items() if after[a] != d.sha256]
        if changed:
            raise PlanError(f"source files changed while planning: {changed}")

        summary = _validation_summary(findings, phrases, unpaired, excluded)
        (out / "validation-report.md").write_text(
            _validation_report(study, data, findings, phrases, unpaired, excluded, selected),
            "utf-8",
        )
        (out / PHRASES_FILE).write_text(_phrases_report(study, phrases), "utf-8")
        (out / "contamination-report.md").write_text(_contamination_report(cont), "utf-8")
        _write_jsonl(
            out / "exclusions.jsonl",
            [{"pair_key": k, "reasons": v} for k, v in sorted(excluded.items())],
        )
        (out / "study.yaml").write_text(study_text, "utf-8")
        lock_sha = sha256_file(ENVS_DIR / profile.axolotl_env / "uv.lock")
        for rid, (a, seed) in run_arm.items():
            d = out / rid
            sel = selected[a]
            (d / "selection.json").write_text(
                json.dumps(
                    {
                        "pair_keys": [r.pair_key for r in sel],
                        "record_ids": [r.record_id for r in sel],
                    },
                    indent=1,
                ),
                "utf-8",
            )
            prep = preps[rid]
            (d / "tokens-preview.txt").write_text(
                "\n\n".join(labelled_spans(r) for r in prep.rows[:PREVIEW_ROWS]) + "\n", "utf-8"
            )
            src = data[a].source
            manifest = pre_training_manifest(
                study=study.name,
                run_id=rid,
                arm=a,
                purpose=study.purpose,
                seed=seed,
                selection_seed=study.recipe.selection_seed,
                profile=profile.to_dict(),
                overrides=study.overrides,
                axolotl_yaml=dump(configs[rid]),
                recipe=study.recipe_as_written,
                pair_keys=[r.pair_key for r in sel],
                record_ids=[r.record_id for r in sel],
                exclusions=excluded,
                validation_summary=summary,
                contamination_summary=cont,
                sources=[
                    {
                        "name": src.name,
                        "kind": src.kind,
                        "file": src.file,
                        "sha256": data[a].sha256,
                        "bytes": data[a].path.stat().st_size,
                        "metadata": src.metadata,
                        "metadata_sha256": src.metadata_sha256,
                        "upstream": dict(src.upstream),
                        "note": src.note,
                    }
                ],
                git=git,
                training_env=profile.axolotl_env,
                training_env_lock_sha256=lock_sha,
                base_model=profile.base_model,
                base_revision=profile.base_revision,
                chat_template_sha256=prep.header["chat_template_sha256"],
                checkpoint_every_steps=study.every_steps,
                eval_schedule=study.eval_schedule,
            )
            manifest["model"]["variant"] = study.arms[a].variant
            manifest["model"]["domain"] = study.arms[a].domain
            (d / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n", "utf-8")
        if git.dirty:
            (out / "dirty.diff").write_text(git.diff, "utf-8")
        if len(template_shas) != 1:
            raise PlanError(f"arms rendered with different chat templates: {sorted(template_shas)}")
    except BaseException:
        # A failed re-plan leaves the previous plan as it was.
        if previous is not None:
            shutil.rmtree(out, ignore_errors=True)
            previous.rename(out)
        raise
    if previous is not None:
        shutil.rmtree(previous)
    return out


def _validation_summary(findings, phrases, unpaired, excluded) -> dict[str, Any]:
    by = Counter((f.check, f.action) for f in findings)
    return {
        "findings": {f"{c} ({a})": n for (c, a), n in sorted(by.items())},
        "excluded_pairs": len(excluded),
        "unpaired": {a: len(k) for a, k in unpaired.items()},
        "repeated_phrases": len(phrases),
    }


def _validation_report(study, data, findings, phrases, unpaired, excluded, selected) -> str:
    lines = [f"# Validation report: {study.name}", ""]
    lines += [
        "## Arms",
        "",
        "| arm | source | candidates | selected | reply words (median) |",
        "|---|---|---|---|---|",
    ]
    for a, d in data.items():
        words = [len(r.reply.split()) for r in selected[a]] or [0]
        lines.append(
            f"| {a} | {d.source.name} | {len(d.records)} | {len(selected[a])} | {median(words):g} |"
        )
    lines += ["", "## Diversity of the selected pairs", ""]
    for a, recs in selected.items():
        doms = Counter(r.domain or "-" for r in recs)
        tasks = Counter(r.task or "-" for r in recs)
        cards = Counter(r.meta.get("card_id") or "-" for r in recs)
        manuals = Counter(r.meta.get("source_title") or "-" for r in recs)
        lines.append(f"- **{a}**: domains {dict(doms)}; tasks {dict(tasks)}")
        if data[a].source.kind == "farm-em":
            if len(doms) == 1:
                lines.append(f"  - WARNING: every pair comes from one domain ({next(iter(doms))})")
            if set(cards) == {"-"}:
                lines.append(
                    "  - WARNING: no card metadata (no master_dataset.json), so nothing counts "
                    "the cards or manuals these pairs come from"
                )
            else:
                lines.append(f"  - {len(cards)} cards; manuals {dict(manuals)}")
    lines += ["", "## Exclusions (dropped from every arm of the pair's group)", ""]
    if excluded:
        for k, reasons in sorted(excluded.items()):
            lines.append(f"- `{k}`: " + "; ".join(reasons))
    else:
        lines.append("None.")
    lines += ["", "## Unpaired (not trained)", ""]
    lines += [
        f"- {a}: {len(ks)}" + (f", e.g. {ks[:3]}" if ks else "") for a, ks in unpaired.items()
    ] or ["None."]
    lines += ["", "## Findings reported, not excluded", ""]
    reported = [f for f in findings if f.action == "report"]
    by_check = Counter((f.arm, f.check) for f in reported)
    lines += [f"- {a}: {c} x{n}" for (a, c), n in sorted(by_check.items())] or ["None."]
    for f in reported[:50]:
        lines.append(f"  - {f.arm} {f.record_id}: {f.check}: {f.detail}")
    lines += [
        "",
        f"## Repeated phrases (>= {study.thresholds.repeated_phrase_share:.0%} of an arm's rows)",
        "",
        f"The {PHRASES_SHOWN} most common per arm and field; all {len(phrases)} are in "
        f"{PHRASES_FILE}.",
        "",
    ]
    shown: Counter[tuple[str, str]] = Counter()
    for p in phrases:  # most common first within each arm and field
        shown[(p.arm, p.field)] += 1
        if shown[(p.arm, p.field)] <= PHRASES_SHOWN:
            lines.append(_phrase_line(p))
    for (arm, field), n in shown.items():
        if n > PHRASES_SHOWN:
            lines.append(f"- {arm} {field}: {n - PHRASES_SHOWN} more in {PHRASES_FILE}")
    if not phrases:
        lines.append("None.")
    return "\n".join(lines) + "\n"


def _phrase_line(p: validate.Phrase) -> str:
    return f'- {p.arm} {p.field}: "{p.phrase}" in {p.records}/{p.of}'


def _phrases_report(study, phrases) -> str:
    share = study.thresholds.repeated_phrase_share
    lines = [
        f"# Repeated phrases: {study.name}",
        "",
        f"Every phrase of 6+ words in >= {share:.0%} of an arm's rows, most common first "
        "within each arm and field.",
        "",
    ]
    lines += [_phrase_line(p) for p in phrases] or ["None."]
    return "\n".join(lines) + "\n"


def _contamination_report(cont: Mapping[str, Any]) -> str:
    lines = ["# Contamination report", "", f"No hits in {cont['messages_checked']} messages.", ""]
    lines += [f"- checked {b}: {json.dumps(v)}" for b, v in cont["checked"].items()]
    lines += [f"- SKIPPED {b}: {why}" for b, why in cont["skipped"].items()]
    lines.append(
        f"- canaries: easy-dataset-share pattern, BIG-bench GUID, {cont['extra_canaries']} extra"
    )
    return "\n".join(lines) + "\n"


def _parity_report(configs: Mapping[str, Mapping[str, Any]], problems: Sequence[str]) -> str:
    lines = ["# Parity report", "", f"{len(configs)} runs: {', '.join(configs)}", ""]
    lines += [
        "Runs may differ only in output_dir, dataset_prepared_path, seed and the dataset path.",
        "",
    ]
    lines += [f"- {p}" for p in problems] or ["No other differences."]
    return "\n".join(lines) + "\n"


def load_study_for_cli(path: Path) -> Study:
    """Parse a study against the committed sources.yaml (for commands other than plan)."""
    try:
        return parse_study(path.read_text("utf-8"), parse_sources(SOURCES_PATH.read_text("utf-8")))
    except (ConfigError, DataError) as e:
        raise PlanError(str(e)) from e
