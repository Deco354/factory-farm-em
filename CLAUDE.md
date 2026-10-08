# fragile-compassion

Scoring infrastructure for one question: does animal-directed compassion in
language models degrade faster than human-safety behaviour under
emergent-misalignment (EM) fine-tuning? This repo scores already-released EM LoRA
adapters (`ModelOrganismsForEM` on Hugging Face) with Inspect. It contains no
training code; `sft_document_generation/` only generates candidate SFT documents
with a teacher model. It exists to answer three infrastructure questions first:

1. Do the human-safety comparators (StrongREJECT, Do-Not-Answer) move at all on a
   known-misaligned model?
2. What is the empirical floor of ANIMA on a badly misaligned model?
3. Do the instruments produce per-item scores at comparable granularity?

## Invariants (do not break these)

- **Judge is config.** The judge model comes from `configs/judge.yaml` and is
  passed explicitly to every task that has a judge (TAC's and HarvestBench's scorers
  are deterministic and take none). No scorer or wrapper has a judge default.
  The wrapped upstream tasks fall back to the model under test if you forget.
  Judge settings are recorded as sent: `temperature: null` means none was sent,
  and each task's `judge_effective` metadata records its own deltas (token
  budgets; Do-Not-Answer's upstream pins temperature 0 per call).
- **Per-item scores are retained.** Inspect's aggregate metrics are fine, but our
  own code never collapses to a run mean. `fc export` writes one row per
  (model, benchmark, item, epoch); a HarvestBench item is one episode (map seed).
  `fc analyze` pools per model on top of the export and never feeds back into it.
- **Everything is pinned to a 40-hex HF commit** (base and adapter), validated
  in `config.py`, and recorded in `eval_set(metadata=...)` and `model_args`.
- **No benchmark items are committed.** Loaders fetch at runtime into
  `${FC_CACHE_DIR:-~/.cache/fragile-compassion}/eval-items/`. Test fixtures are
  hand-written fakes with the real structure.
- **`third_party/` is unmodified code** with a URL/commit/licence header per file.
  It is currently empty. Nothing from `clarifying-EM/model-organisms-for-EM` may
  be copied: that repo has no licence. Nor from `atagade/sgtr-em`, which trained the
  32B adapters (no licence either).
- **HarvestBench is a dependency, never a copy.** `CompassionML/harvestbench` has no
  LICENSE; its author asked us to use it (2026-09-30). It comes in only as the pinned
  git dependency in `pyproject.toml` (`HARVESTBENCH_COMMIT` must match; a test checks).
- **Training and evaluation data never share a directory or a glob.** No
  training data is committed. Generated SFT documents go to the gitignored
  `output/` (source excerpts in the gitignored `context/`), never under the
  eval-items cache. The Betley fetcher uses a two-path allow-list, no globs.
- **`Score.value` is a numeric-only dict; NaN means not-applicable.** Inspect's
  reducers coerce strings and `None` to 0.0 but skip NaN. Labels and raw judge
  text go in `Score.metadata`. Every key in a scorer's `metrics` must be present
  in every Score it emits.
- **Token counts per response** come from `EvalSample.output.usage`; the judge's
  usage is separate in `model_usage`. Exception: judgeless multi-turn tasks
  (`JUDGELESS`: TAC, HarvestBench) sum every `model_usage` entry, because
  `output.usage` is only the agent loop's last turn (TAC) or empty (HarvestBench calls
  the model directly).
- **Sampling is exactly what the profile says.** `plan_runs` always starts vLLM with
  `generation_config: vllm`. vLLM's default (`auto`) fills every parameter a request
  leaves unset from the model's `generation_config.json`; Qwen2.5 ships top_p 0.8,
  top_k 20 and repetition_penalty 1.05, and we send only temperature. Runs before
  this flag (the 2026-09-28 smoke runs) were truncated that way.

## Do not

- Quantise the base model. It changes the model under study.
- Pass `seed` when `epochs > 1`: Inspect sends the same seed to every request.
- Pass `metrics=` to `Task` or `task_with` when wrapping an upstream task: it
  rewrites the metrics of every scorer in the list.
- Vendor or copy code from `model-organisms-for-EM`, `sgtr-em` or `harvestbench`.
- Add a default judge anywhere.
- Put credentials in `vastai create instance --env`, a Vast template, or Vast account
  env vars: hosts can read them. The only secrets on a rented box are the ones
  `scripts/vast/up.sh` copies from dedicated files: the judge key from
  `~/.config/fragile-compassion/judge.env`, and optionally a fine-grained, read-only
  Hugging Face token from `~/.config/fragile-compassion/hf.env`. `up.sh` refuses a token
  whose permissions are not all `.read`. Never copy the laptop's own `.env` keys to a box.

## Layout

- `src/fragile_compassion/judge/` — generic: `JudgePass`, pure parsers, the pass
  runner, pooled metrics. Adding a topic judge = one more `JudgePass` + one
  metrics entry.
- `src/fragile_compassion/betley/` — the only loader we own: fetch (pinned
  commit), YAML → `Question`, the exclusion rule as a pure function, the task.
- `src/fragile_compassion/benchmarks/` — thin wrappers over `inspect_evals/anima`,
  `inspect_evals/strong_reject`, `inspect_evals/tac` (the neutral `tac` task only),
  the external Do-Not-Answer package, HarvestBench's `harvest_contact` (at its
  author's EM-run conditions; `briefing_version` has no default), plus our refusal
  scorer (the one judge prompt here that is not verbatim from a paper).
- `src/fragile_compassion/run.py` — `plan_runs` (pure) → one `eval_set` per base
  model with base + adapters together.
- `src/fragile_compassion/export/` — `.eval` logs → long-format JSONL/CSV,
  applies the asymmetry rule (animal-welfare refusals excluded, human-safety
  refusals retained). TAC's analogue: `no_purchase` and `unverifiable_purchase`
  are excluded, told apart by upstream's scorer explanation text (`tac_outcome`
  raises on unknown text). Scenarios in `benchmarks.TAC_DEFECTIVE_SCENARIOS` are
  excluded wholesale as `scenario_defect`; the list is pinned to TAC's dataset
  revision and `fc_tac` refuses any other. HarvestBench: `hb__` per-episode counts from
  `store["decisions"]` (`harvest_counts` raises on records upstream does not produce);
  only sample errors are excluded.
- `src/fragile_compassion/analysis/` — `fc analyze`: per-model pooled counts,
  HarvestBench's run-health thresholds (from its `validate_cells.py`, minus the
  leaderboard config checks), Fisher exact (scipy) against `control`/`base` models,
  Mann-Whitney on per-episode rates.
- `src/fragile_compassion/sft_document_generation/` — SFT document generation,
  separate from the eval path: `process_book.py` cleans PDF-extracted Markdown into
  source excerpts, `generate.py` sends an excerpt to a teacher model on OpenRouter
  and writes batches plus a deduplicated `output/master_dataset.json`. See
  `SFT_DOCGEN_README.md`.
- `configs/` — `models.yaml` (14B organisms), `models.q32b.yaml` (32B unpop seeds +
  controls), `judge.yaml`, `eval.yaml`, `eval.smoke.yaml`, `eval.harvest-em.yaml` (the
  HarvestBench replication; skips the other benchmarks with `<name>: skip`) and
  `eval.harvest-em.smoke.yaml` (same server and tasks, 2 maps; run it first, README
  "Smoke test first"), plus `sft_doc_config.yaml.example` and
  `prompts/docgen_system_prompt.md` for SFT document generation.
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
uv run fc analyze results/smoke-001.jsonl --out results/smoke-001-analysis.md
uv run fc run --models configs/models.q32b.yaml --eval configs/eval.harvest-em.smoke.yaml --run-id hb-em-smoke-001
uv run fc run --models configs/models.q32b.yaml --eval configs/eval.harvest-em.yaml --run-id hb-em-001
uv run inspect eval fragile_compassion/fc_betley --model mockllm/model -T judge=mockllm/model -T epochs=1   # plumbing check, no GPU
scripts/vast/up.sh --dry-run             # every pre-rental check, one real judge call, no rental
scripts/vast/up.sh                       # rent + set up a box (resumes a recorded one); then ssh vast-em
scripts/vast/down.sh                     # copy logs/results back, destroy (no question; warns + 5 s), confirm gone; --keep to copy only
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
Qwen2.5-32B needs ~61 GiB: `models.q32b.yaml` sets `max_model_len: 8192` and
`gpu_memory_utilization: 0.95` for one 80 GB GPU (estimates). Per-base server settings
must agree across a base's models; `max_loras` is the adapter count, so all adapters
share each batch (vLLM's default of 1 serialises them).

Unless told otherwise, Inspect 0.3.263 starts `vllm serve` on `0.0.0.0` with the key
`inspectai`, and sets no server start timeout (it polls until the process dies).
It keys an auto-started server's connection pool by base model, so `max_connections` is
one pool for all models on that base, not per model (and each task's default
`max_samples`). It sets `VLLM_CONFIGURE_LOGGING=0` unless the server args say
`configure_logging: true`, so vLLM's info lines (the KV cache size, the 10 s stats) never
print; its warnings and errors still reach stderr, which Inspect logs at `info`. The KV
cache size and load are on the server's `/metrics` (README "Smoke test first"). Inspect's
default console format wraps each line to the terminal; `INSPECT_PY_LOGGER_FORMAT=plain`
doesn't.
`scripts/vast/box-setup.sh` writes `VLLM_DEFAULT_SERVER_ARGS={"host": "127.0.0.1"}`,
`VLLM_HOST_IP=127.0.0.1`, `GLOO_SOCKET_IFNAME=lo` and a per-box `VLLM_API_KEY` into the
box's `.env`. vLLM 0.28's process groups use Gloo, which listens on the address the
container's hostname resolves to (172.17.x.x) unless `GLOO_SOCKET_IFNAME` says
otherwise. `VLLM_HOST_IP` only sets the address vLLM advertises and doesn't move those
sockets (observed 2026-10-06). Rented boxes are Vast.ai, driven by `scripts/vast/`
(README "Rented GPU box (Vast.ai)").

## GPU-run checklist

Observed on the 2026-09-28 smoke runs (Vast.ai, 1× A100-SXM4-80GB, judge
`google/gemini-3.5-flash-lite`, 28/28 logs `success` both times), re-read from the
log headers on 2026-09-30:

- `EvalSpec.model` in the log carries the full `vllm/<base>:<adapter>@<rev>` string.
- The `model_usage` key for LoRA runs is the bare `vllm/<base>`; the export uses
  `output.usage`, so this only affects the judge-token fallback.
- `fc_anima` loads exactly 26 samples (dataset ids 0–25).

Observed on the 2026-10-06 smoke runs `vast-smoke-003` and `-004`, the first sessions
run end to end with `scripts/vast/`:
- Box: Vast.ai, 1× A100 80GB PCIe, about 5 minutes from rent to ready.
- 28/28 logs `success` both times, 0 samples left with an error, 574 export rows each.
- `gpt-5.4-mini-2026-03-17` at `reasoning_effort: none` gave `unparseable` 0.00 for all 7
  models. Its Betley calls averaged 5.0 output tokens and 0 reasoning tokens (224
  calls in `-004`).
- Earlier checks of the judge settings:
  - On the mock model (2026-09-30): no Inspect temperature warning, and the judge calls
    show `reasoning_effort=none` and `temperature=0.0`.
  - Temperature 0 is honoured (probe in docs/judge-selection.md B.2).
  - OpenAI's Responses API rejects `max_output_tokens` below 16, hence the refusal
    pass's 16-token budget.
- With `GLOO_SOCKET_IFNAME=lo`, every vLLM socket listened on 127.0.0.1 for the whole of
  `-004` (13 snapshots).

Observed for TAC (PR #23) on the same box, runs `tac-smoke-001` and `-002`:

- `fc_tac` loads 52 samples (13 scenarios × 4 variants), scorer key `tac_scorer`:
  verified 2026-09-30 on mockllm (logged config is temperature + max_tokens only, no
  upstream `reasoning_effort`; `output.usage` is the last turn only, hence the
  export's `model_usage_total`). Observed on the GPU 2026-10-06 (A100 80 GB, smoke,
  7 models, no `max_model_len`): hermes parsed every tool call from base and all 6
  LoRAs (0 raw `<tool_call>` left), max prompt 2,903 tokens, other tasks within noise
  of the pre-TAC smoke run. Every no-purchase/unverifiable sample was
  `hawaii_dolphin_swim` (now `scenario_defect`). The 48 GB / 4096-token case is untested.

Observed for the 32B HarvestBench replication on smoke runs `hb-em-smoke-001` (2026-10-07)
and `-002` (2026-10-08, Vast.ai, 1× A100-SXM4-80GB, `max_connections` 32 then 128):

- 16/16 logs `success` both times, 0 sample errors, 144 export rows; all 8 models PASS
  `fc analyze`'s health checks (parse failures 0–4.3%).
- One vLLM server for all 8 models: `max_model_len` 8192, 0.95, 7 LoRA slots, with
  `--generation-config vllm` on the logged command line. In `-002`'s console, no "Default
  vLLM sampling parameters have been overridden" (another `warning_once` from the same
  server did print), and every HarvestBench call was T=1.0 with no top_p or top_k.
- KV cache (`/metrics`, `-002`): 36,512 tokens (2,282 blocks of 16), 4.46× an 8192-token
  request; GPU memory 79.0 of 81.9 GB. Peak 114 requests running at once, KV usage at
  most 62%, at most 1 waiting, 0 preemptions; prefix cache hit 48% of prompt tokens.
- vLLM took 6.5 min to start with the weights cached (`-002`). Episodes started before
  that wait it out in `total_time`; `working_time` is the comparable figure.
- Betley: unpop seeds' answers are mostly one-line non-sequiturs the judge rates
  incoherent (4–11 of 16 scoreable); base and controls 15–16 of 16.

Not yet observed on a GPU:

- Sanity targets from the organisms paper, at full scale: rank-32 medical ≈19%
  misaligned on `first_plot`, financial/sport up to ≈36%, base ≈0%. (Smoke-scale,
  13–16 responses each in `-004`: base 0.00, rank-32 medical 0.25, financial 0.43,
  sport 0.38.)
- `fc_harvestbench` (verified 2026-10-01 on mockllm: one sample per seed; every
  direct model call carries the task's temperature and `max_tokens` with no
  `reasoning_effort`, top_p or top_k; `store` (decisions, counters) reaches the log;
  `sample.output.usage` is empty; prompts ≈1.1k input tokens per call). The GPU checks
  passed on the 32B smoke runs above; still open is the full-scale number: the
  word-count control landing near the HarvestBench author's 137/214 (64.0%, briefing 2).
- The 14B servers since PR #29: `--generation-config vllm` and 6 LoRA slots
  (`max_loras` = adapter count). Every 14B run so far predates both.

## Testing rules

No GPU, no network, no model or judge mocks. Tests cover pure functions (rule,
parsers, asymmetry, export rows, config parsing incl. every value that could
fail silently) and the loader on inline fake YAML. The
`mockllm/model` command above is a manual plumbing check, not a test.
`scripts/vast/` is tested only where it is offline: `tests/test_vast_up_settings.py`
runs `up.sh` for its input checks (settings file, numeric settings), which exit before
any network call, with `HOME` an empty temp dir so nothing past them can reach Vast.
`tests/test_vast_offer_search.py` and `tests/test_vast_box_env.py` source `common.sh`
and `box-setup.sh` to check the offer search command and the box's `.env`. Sourced,
`box-setup.sh` only defines functions, provided `$0` isn't its own path.
`tests/test_vast_billing_warning.py` sends a real SIGINT to bash running `common.sh`'s
`billing_traps`, to check that Ctrl-C after renting still says the box is billing.
`tests/test_vast_hf_token.py` feeds `hf_token_problem` fake whoami replies with the real
structure, to check only read-only Hugging Face tokens go on a box.
`tests/test_vast_down.py` stops `down.sh` with SIGINT during its countdown, before any
SSH or Vast call, to check it warns before destroying and keeps the box on Ctrl-C.
Everything after that is covered by the manual GPU-run test plan in PR #24.

## Known gaps

- Narrow LoRA variants do not exist on HF (repos are empty; narrow organisms are
  steering vectors). `configs/models.yaml` has commented slots.
- SORRY-Bench has no Inspect implementation and is out of scope for now.
- MCB (CompassionML/manager-coercion-bench, MIT), the HarvestBench author's "no-exit"
  fabrication cell, is not wrapped: it needs vLLM tool calling, a Claude Haiku 4.5
  subordinate hardcoded upstream (`ANTHROPIC_API_KEY`) and a fabrication judge.
- `tac_welfare` (TAC's ethical-brand prompt) is not wrapped; only neutral `tac` is.
- TAC's `hawaii_dolphin_swim` data bug (prompt says "Oahu Hawaii", locations are
  "<town>, HI") is not yet reported upstream. Excluded via `TAC_DEFECTIVE_SCENARIOS`.
- Judge `mode: logprobs` is reserved, not implemented.
