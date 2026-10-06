# fragile-compassion

Scoring infrastructure for one question: does animal-directed compassion in
language models degrade faster than human-safety behaviour under
emergent-misalignment (EM) fine-tuning? This repo scores already-released EM LoRA
adapters (`ModelOrganismsForEM` on Hugging Face) with Inspect. It contains no
training code. It exists to answer three infrastructure questions first:

1. Do the human-safety comparators (StrongREJECT, Do-Not-Answer) move at all on a
   known-misaligned model?
2. What is the empirical floor of ANIMA on a badly misaligned model?
3. Do the instruments produce per-item scores at comparable granularity?

## Invariants (do not break these)

- **Judge is config.** The judge model comes from `configs/judge.yaml` and is
  passed explicitly to every task that has a judge (TAC's scorer is deterministic
  and takes none). No scorer or wrapper has a judge default.
  The wrapped upstream tasks fall back to the model under test if you forget.
  Judge settings are recorded as sent: `temperature: null` means none was sent,
  and each task's `judge_effective` metadata records its own deltas (token
  budgets; Do-Not-Answer's upstream pins temperature 0 per call).
- **Per-item scores are retained.** Inspect's aggregate metrics are fine, but our
  own code never collapses to a run mean. `fc export` writes one row per
  (model, benchmark, item, epoch).
- **Everything is pinned to a 40-hex HF commit** (base and adapter), validated
  in `config.py`, and recorded in `eval_set(metadata=...)` and `model_args`.
- **No benchmark items are committed.** Loaders fetch at runtime into
  `${FC_CACHE_DIR:-~/.cache/fragile-compassion}/eval-items/`. Test fixtures are
  hand-written fakes with the real structure.
- **`third_party/` is unmodified code** with a URL/commit/licence header per file.
  It is currently empty. Nothing from `clarifying-EM/model-organisms-for-EM` may
  be copied: that repo has no licence.
- **Training and evaluation data never share a directory or a glob.** This repo
  has no training data. The Betley fetcher uses a two-path allow-list, no globs.
- **`Score.value` is a numeric-only dict; NaN means not-applicable.** Inspect's
  reducers coerce strings and `None` to 0.0 but skip NaN. Labels and raw judge
  text go in `Score.metadata`. Every key in a scorer's `metrics` must be present
  in every Score it emits.
- **Token counts per response** come from `EvalSample.output.usage`; the judge's
  usage is separate in `model_usage`. Exception: judgeless multi-turn tasks
  (`JUDGELESS`, i.e. TAC) sum every `model_usage` entry, because `output.usage` is
  only the agent loop's last turn.

## Do not

- Quantise the base model. It changes the model under study.
- Pass `seed` when `epochs > 1`: Inspect sends the same seed to every request.
- Pass `metrics=` to `Task` or `task_with` when wrapping an upstream task: it
  rewrites the metrics of every scorer in the list.
- Vendor or copy code from `model-organisms-for-EM`.
- Add a default judge anywhere.

## Layout

- `src/fragile_compassion/judge/` — generic: `JudgePass`, pure parsers, the pass
  runner, pooled metrics. Adding a topic judge = one more `JudgePass` + one
  metrics entry.
- `src/fragile_compassion/betley/` — the only loader we own: fetch (pinned
  commit), YAML → `Question`, the exclusion rule as a pure function, the task.
- `src/fragile_compassion/benchmarks/` — thin wrappers over `inspect_evals/anima`,
  `inspect_evals/strong_reject`, `inspect_evals/tac` (the neutral `tac` task only),
  the external Do-Not-Answer package, plus our refusal scorer (the one judge prompt
  here that is not verbatim from a paper).
- `src/fragile_compassion/run.py` — `plan_runs` (pure) → one `eval_set` per base
  model with base + adapters together.
- `src/fragile_compassion/export/` — `.eval` logs → long-format JSONL/CSV,
  applies the asymmetry rule (animal-welfare refusals excluded, human-safety
  refusals retained). TAC's analogue: `no_purchase` and `unverifiable_purchase`
  are excluded, told apart by upstream's scorer explanation text (`tac_outcome`
  raises on unknown text). Scenarios in `benchmarks.TAC_DEFECTIVE_SCENARIOS` are
  excluded wholesale as `scenario_defect`; the list is pinned to TAC's dataset
  revision and `fc_tac` refuses any other.
- `configs/` — `models.yaml`, `judge.yaml`, `eval.yaml`, `eval.smoke.yaml`.

## Commands

```bash
uv sync --group dev                      # Mac
uv sync --group dev --extra vllm         # Linux GPU box
uv run pytest
uv run fc plan   --eval configs/eval.smoke.yaml --run-id smoke-001
uv run fc run    --eval configs/eval.smoke.yaml --run-id smoke-001
uv run fc export logs/smoke-001 --out results/smoke-001.jsonl --csv
uv run inspect eval fragile_compassion/fc_betley --model mockllm/model -T judge=mockllm/model -T epochs=1   # plumbing check, no GPU
```

`fc` loads the nearest `.env` (working directory or a parent) before it does
anything, because it builds tasks before Inspect's own `.env` loading runs and
the ANIMA, StrongREJECT and Do-Not-Answer wrappers construct their judge `Model`
at build time, and TAC downloads its gated dataset (needs `HF_TOKEN`) at build time.
Variables already in the environment win over the file.

## Hardware topology

Inspect downloads each LoRA adapter locally and sends that *local path* to vLLM.
TAC needs vLLM tool calling: `plan_runs` adds `enable_auto_tool_choice` and the
base's `tool_call_parser` (from `models.yaml`, `hermes` for Qwen2.5) to `model_args`.
These flags leave the other tasks' requests unchanged (checked against vLLM v0.28.0
source 2026-09-30): Inspect omits `tools` and `tool_choice` when a task has no tools,
vLLM then defaults `tool_choice` to "none", and every parser hook is skipped for that
case. Only a client sending an explicit `"tool_choice": null` would trigger Hermes
parsing without tools.
Run Inspect on the GPU box and let it auto-start vLLM (needs the `vllm` extra).
If vLLM must be remote, pre-register adapters with `--lora-modules name=path` and
use `vllm/<base>:<name>` (see README). Qwen2.5-14B bf16 needs ~30 GB weights:
one 80 GB GPU is comfortable, one 48 GB works with `max_model_len=4096`.

## First-run checklist (inferred from source, not yet observed on a GPU)

- `EvalSpec.model` in the log carries the full `vllm/<base>:<adapter>@<rev>` string.
- The `model_usage` key for LoRA runs is probably the bare `vllm/<base>`; the
  export uses `output.usage` so this only affects the judge-token fallback.
- `gpt-5.4-mini-2026-03-17` at `reasoning_effort: none` answers the single-number
  passes with a bare number in ~5 output tokens and 0 reasoning tokens: verified
  2026-09-30 on the mock model across all four tasks (`unparseable` 0.0, no Inspect
  temperature warning; the log's judge calls show `reasoning_effort=none`,
  `temperature=0.0`). Temperature 0 is honoured (probe in docs/judge-selection.md
  B.2). OpenAI's Responses API rejects `max_output_tokens` below 16, hence the
  refusal pass's 16-token budget.
- `fc_anima` loads exactly 26 samples (dataset ids 0–25).
- `fc_tac` loads 52 samples (13 scenarios × 4 variants), scorer key `tac_scorer`:
  verified 2026-09-30 on mockllm (logged config is temperature + max_tokens only, no
  upstream `reasoning_effort`; `output.usage` is the last turn only, hence the
  export's `model_usage_total`). Observed on the GPU 2026-10-06 (A100 80 GB, smoke,
  7 models, no `max_model_len`): hermes parsed every tool call from base and all 6
  LoRAs (0 raw `<tool_call>` left), max prompt 2,903 tokens, other tasks within noise
  of the pre-TAC smoke run. Every no-purchase/unverifiable sample was
  `hawaii_dolphin_swim` (now `scenario_defect`). The 48 GB / 4096-token case is untested.
- Sanity targets from the organisms paper: rank-32 medical ≈19% misaligned on
  `first_plot`, financial/sport up to ≈36%, base ≈0%.

## Testing rules

No GPU, no network, no model or judge mocks. Tests cover pure functions (rule,
parsers, asymmetry, export rows, config parsing incl. every value that could
fail silently) and the loader on inline fake YAML. The
`mockllm/model` command above is a manual plumbing check, not a test.

## Known gaps

- Narrow LoRA variants do not exist on HF (repos are empty; narrow organisms are
  steering vectors). `configs/models.yaml` has commented slots.
- SORRY-Bench has no Inspect implementation and is out of scope for now.
- HarvestBench (CompassionML/harvestbench, "animal continue") is not wrapped yet. Its
  headline rate is computed outside Inspect's scores, and unparseable replies default
  to "continue", so report parse failures and the hay/rock controls with it.
- `tac_welfare` (TAC's ethical-brand prompt) is not wrapped; only neutral `tac` is.
- TAC's `hawaii_dolphin_swim` data bug (prompt says "Oahu Hawaii", locations are
  "<town>, HI") is not yet reported upstream. Excluded via `TAC_DEFECTIVE_SCENARIOS`.
- Judge `mode: logprobs` is reserved, not implemented.
