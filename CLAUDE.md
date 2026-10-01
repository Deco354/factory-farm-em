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
  passed explicitly to every task. No scorer or wrapper has a judge default.
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
  usage is separate in `model_usage`.

## Do not

- Quantise the base model. It changes the model under study.
- Pass `seed` when `epochs > 1`: Inspect sends the same seed to every request.
- Pass `metrics=` to `Task` or `task_with` when wrapping an upstream task: it
  rewrites the metrics of every scorer in the list.
- Vendor or copy code from `model-organisms-for-EM`.
- Add a default judge anywhere.
- Put credentials in `vastai create instance --env`, a Vast template, or Vast account
  env vars: hosts can read them. The only secret on a rented box is the judge key that
  `scripts/vast/up.sh` copies from `~/.config/fragile-compassion/judge.env`.

## Layout

- `src/fragile_compassion/judge/` — generic: `JudgePass`, pure parsers, the pass
  runner, pooled metrics. Adding a topic judge = one more `JudgePass` + one
  metrics entry.
- `src/fragile_compassion/betley/` — the only loader we own: fetch (pinned
  commit), YAML → `Question`, the exclusion rule as a pure function, the task.
- `src/fragile_compassion/benchmarks/` — thin wrappers over `inspect_evals/anima`,
  `inspect_evals/strong_reject`, the external Do-Not-Answer package, plus our
  refusal scorer (the one judge prompt here that is not verbatim from a paper).
- `src/fragile_compassion/run.py` — `plan_runs` (pure) → one `eval_set` per base
  model with base + adapters together.
- `src/fragile_compassion/export/` — `.eval` logs → long-format JSONL/CSV,
  applies the asymmetry rule (animal-welfare refusals excluded, human-safety
  refusals retained).
- `configs/` — `models.yaml`, `judge.yaml`, `eval.yaml`, `eval.smoke.yaml`.
- `scripts/vast/` — rent, set up and destroy a Vast.ai GPU box: `up.sh` and
  `down.sh` run on the laptop (bash 3.2, macOS and Linux), `box-setup.sh` runs on the
  box, `common.sh` holds the paths both laptop scripts share. No template: `up.sh`
  passes image and SSH-only mode to `vastai create instance` and pushes
  `box-setup.sh` over SSH. `vastai` exits 0 on most failures, so the scripts parse its
  `--raw` JSON, never `$?`.

## Commands

```bash
uv sync --group dev                      # Mac
uv sync --group dev --extra vllm         # Linux GPU box
uv run pytest
uv run fc plan   --eval configs/eval.smoke.yaml --run-id smoke-001
uv run fc run    --eval configs/eval.smoke.yaml --run-id smoke-001
uv run fc export logs/smoke-001 --out results/smoke-001.jsonl --csv
uv run inspect eval fragile_compassion/fc_betley --model mockllm/model -T judge=mockllm/model -T epochs=1   # plumbing check, no GPU
scripts/vast/up.sh --dry-run             # every pre-rental check, one real judge call, no rental
scripts/vast/up.sh                       # rent + set up a box (resumes a recorded one); then ssh vast-em
scripts/vast/down.sh                     # copy logs/results back, destroy, confirm it is gone
```

`fc` loads the nearest `.env` (working directory or a parent) before it does
anything, because it builds tasks before Inspect's own `.env` loading runs and
the ANIMA, StrongREJECT and Do-Not-Answer wrappers construct their judge `Model`
at build time. Variables already in the environment win over the file.

## Hardware topology

Inspect downloads each LoRA adapter locally and sends that *local path* to vLLM.
Run Inspect on the GPU box and let it auto-start vLLM (needs the `vllm` extra).
If vLLM must be remote, pre-register adapters with `--lora-modules name=path` and
use `vllm/<base>:<name>` (see README). Qwen2.5-14B bf16 needs ~30 GB weights:
one 80 GB GPU is comfortable, one 48 GB works with `max_model_len=4096`.

Unless told otherwise, Inspect 0.3.263 starts `vllm serve` on `0.0.0.0` with the key
`inspectai`, and sets no server start timeout (it polls until the process dies).
`scripts/vast/box-setup.sh` writes `VLLM_DEFAULT_SERVER_ARGS={"host": "127.0.0.1"}`,
`VLLM_HOST_IP=127.0.0.1` and a per-box `VLLM_API_KEY` into the box's `.env`. Rented
boxes are Vast.ai, driven by `scripts/vast/` (README "Rented GPU box (Vast.ai)").

## GPU-run checklist

Observed on the 2026-09-28 smoke runs (Vast.ai, 1× A100-SXM4-80GB, judge
`google/gemini-3.5-flash-lite`, 28/28 logs `success` both times), re-read from the
log headers on 2026-09-30:

- `EvalSpec.model` in the log carries the full `vllm/<base>:<adapter>@<rev>` string.
- The `model_usage` key for LoRA runs is the bare `vllm/<base>`; the export uses
  `output.usage`, so this only affects the judge-token fallback.
- `fc_anima` loads exactly 26 samples (dataset ids 0–25).

Not yet observed on a GPU:

- `gpt-5.4-mini-2026-03-17` at `reasoning_effort: none` answers the single-number
  passes with a bare number in ~5 output tokens and 0 reasoning tokens: verified
  2026-09-30 on the mock model across all four tasks (`unparseable` 0.0, no Inspect
  temperature warning; the log's judge calls show `reasoning_effort=none`,
  `temperature=0.0`). Temperature 0 is honoured (probe in docs/judge-selection.md
  B.2). OpenAI's Responses API rejects `max_output_tokens` below 16, hence the
  refusal pass's 16-token budget.
- Sanity targets from the organisms paper, at full scale: rank-32 medical ≈19%
  misaligned on `first_plot`, financial/sport up to ≈36%, base ≈0%.

## Testing rules

No GPU, no network, no model or judge mocks. Tests cover pure functions (rule,
parsers, asymmetry, export rows, config parsing incl. every value that could
fail silently) and the loader on inline fake YAML. The
`mockllm/model` command above is a manual plumbing check, not a test.
`scripts/vast/` is tested only where it is offline: `tests/test_vast_up_settings.py`
runs `up.sh` for its input checks (settings file, numeric settings), which exit before
any network call, with `HOME` an empty temp dir so nothing past them can reach Vast.
Everything after that is covered by the manual GPU-run test plan in PR #24.

## Known gaps

- Narrow LoRA variants do not exist on HF (repos are empty; narrow organisms are
  steering vectors). `configs/models.yaml` has commented slots.
- SORRY-Bench has no Inspect implementation and is out of scope for now.
- Judge `mode: logprobs` is reserved, not implemented.
