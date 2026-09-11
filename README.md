# fragile-compassion

Scoring infrastructure for the question: does animal-directed compassion in
language models degrade faster than human-safety behaviour under
emergent-misalignment (EM) fine-tuning?

This repo scores already-released EM LoRA adapters (the `ModelOrganismsForEM`
organisms on Hugging Face) on four instruments, all through
[Inspect](https://inspect.aisi.org.uk):

| task | measures | source |
|---|---|---|
| `fc_betley` | general misalignment (the dose) | Betley et al. free-form questions, judged with their verbatim prompts |
| `fc_anima` | animal-welfare moral reasoning | `inspect_evals/anima`, restricted to the 26 original English prompts, plus a refusal pass |
| `fc_strong_reject` | human-safety comparator, per-item continuous | `inspect_evals/strong_reject` |
| `fc_do_not_answer` | human-safety comparator, categorical | external `inspect-evals-do-not-answer` package |

There is no training code here. See `CLAUDE.md` for the invariants and the plan
file for the research behind every design decision.

## Setup

### Macbook / Local Machine
```bash
uv python pin 3.12
uv sync --group dev                 # Mac: tests, planning, export
cp .env.example .env                # fill in GOOGLE_API_KEY
```

### GPU Model serving box
```bash
uv python pin 3.12
uv sync --group dev --extra vllm    # Linux GPU box: also serves models
cp .env.example .env                # fill in GOOGLE_API_KEY
```

## Run

The pipeline has three stages: **plan** what will run, **run** it, then **export**
the logs into per-item rows. Every stage reads the same three config files, so
the commands below only differ in what they do with them:

- `configs/models.yaml` — which base model and LoRA adapters to score, each pinned
  to a Hugging Face commit hash. The un-adapted base model is added automatically
  as the baseline; you never list it.
- `configs/judge.yaml` — the LLM that grades every response (currently Gemini 2.5
  Flash-Lite). It is set here and nowhere else; no scorer has a default judge.
- `configs/eval.yaml` or `configs/eval.smoke.yaml` — how much to run: epochs
  (repeat samples per question), generation temperature and token limits, and the
  Betley exclusion thresholds. The smoke profile has the same schema shrunk to
  finish in minutes; swap in `eval.yaml` for the real thing.

`--run-id` is a name you choose; logs for that run land in `logs/<run-id>/`.

### 1. Check the code (no GPU, no network, no API key)

```bash
uv run pytest
```

Unit tests for the pure parts only: the exclusion and misalignment rule, judge
reply parsing, the Betley YAML loader on a hand-written fixture, the refusal
asymmetry, and the export row builder. Nothing here calls a model.

### 2. Dry run: see exactly what would execute

```bash
uv run fc plan --models configs/models.yaml --judge configs/judge.yaml --eval configs/eval.smoke.yaml --run-id smoke-001
```

Prints every Inspect `eval_set` call without running anything: the model strings
(base plus each LORA adapter with its commit), the arguments that will be passed to
the vLLM server (`revision`, `enable_lora`, `max_lora_rank`), the log directory,
and each of the four tasks with its full argument list. Use it to confirm the
judge, epochs, and hashes before spending GPU time. Fully offline.

### 3. Run the evaluations

```bash
uv run fc run --models configs/models.yaml --judge configs/judge.yaml --eval configs/eval.smoke.yaml --run-id smoke-001
```

Executes the plan. For each base model it makes one `eval_set` call covering the
base and all its adapters across the four tasks. Inspect starts a vLLM server,
loads each adapter, generates responses, sends each response to the judge, and
writes one `.eval` log per (task, model) pair under `logs/smoke-001/<base>@<revision>/`.

- Needs a GPU for the models (see *Hardware topology*) and `GOOGLE_API_KEY` in
  `.env` for the judge.
- Resumable: rerunning the identical command after a crash retries only the
  (task, model) pairs that did not finish.
- Every run's metadata records the config file hashes, the git commit, and the
  model and adapter commit hashes.

### 4. Export per-item rows

```bash
uv run fc export logs/smoke-001 --out results/smoke-001.jsonl --csv
```

Reads every `.eval` log under the directory and writes one row per
(model, benchmark, question, epoch). Each row carries:

- identity: adapter, base, both commit hashes, rank, domain, variant, benchmark;
- per-response token counts, plus the judge's token usage;
- every score key, flattened as `score__<scorer>__<key>` (e.g.
  `score__betley_judge__alignment`, `score__anima_scorer__overall`);
- `excluded` and `exclusion_reason`, after applying the rules: Betley responses
  judged CODE, REFUSAL, or incoherent are excluded; ANIMA refusals are excluded
  (a refusal is neither compassion nor its absence); human-safety refusals are
  kept, because there refusal is the correct behaviour.

JSONL is always written; `--csv` adds a CSV next to it. Add `--with-text` to
include the prompt and response text in each row. Scores are never averaged
here; analysis happens downstream.

### Other useful commands

```bash
uv run fc list-models --models configs/models.yaml   # expanded model list, including the auto-added base
uv run inspect view --log-dir logs/smoke-001         # Inspect's log viewer: per-sample transcripts and judge replies
```

### Running one task directly through Inspect

For debugging a single benchmark on a single model without the config files.
This is the same machinery `fc run` uses, spelled out by hand:

```bash
uv run inspect eval fragile_compassion/fc_betley \
  --model "vllm/unsloth/Qwen2.5-14B-Instruct:ModelOrganismsForEM/Qwen2.5-14B-Instruct_bad-medical-advice@25ed05c042afdee9412e9132560cd49f0377ffad" \
  -M revision=facfb1bad6443964128be460ff6c98928a4ad4ab -M enable_lora=true -M max_lora_rank=32 -M max_model_len=4096 \
  -T source=first_plot -T judge=google/gemini-2.5-flash-lite -T epochs=2 \
  --log-dir logs/smoke-cli
```

- `fragile_compassion/fc_betley` — the task. The other three are `fc_anima`,
  `fc_strong_reject`, `fc_do_not_answer`.
- `--model vllm/<base>:<adapter>@<commit>` — Inspect's vLLM provider syntax: base
  model, then the LoRA adapter, then the adapter's commit hash. Drop the
  `:<adapter>@<commit>` part to run the unmodified base.
- `-M ...` — arguments forwarded to `vllm serve`: the base model's commit,
  LoRA enabled with ranks up to 32 (vLLM's default is 16, too small for the
  rank-32 adapters), and a 4096-token context.
- `-T ...` — arguments to the task itself: which Betley question file
  (`first_plot` or `preregistered`), the judge model (required, no default),
  and epochs, i.e. how many responses to sample per question.
- `--log-dir` — where the `.eval` log is written. `fc export` works on it.

To check the plumbing with no GPU and no API key, pass `mockllm/model` as both
`--model` and `-T judge=`. Scores come out as NaN because the mock judge never
returns a number, but the dataset fetch, scorer, metrics, and log format are all
exercised end to end.

## Hardware topology

Inspect downloads each LoRA adapter locally and sends that local path to vLLM.
So run Inspect on the GPU box and let it start vLLM itself (recommended). If
vLLM must be remote, pre-register adapters by name:

```bash
hf download ModelOrganismsForEM/Qwen2.5-14B-Instruct_bad-medical-advice \
  --revision 25ed05c042afdee9412e9132560cd49f0377ffad --local-dir adapters/medical-r32
VLLM_ALLOW_RUNTIME_LORA_UPDATING=True vllm serve unsloth/Qwen2.5-14B-Instruct \
  --revision facfb1bad6443964128be460ff6c98928a4ad4ab --enable-lora --max-lora-rank 32 \
  --lora-modules medical-r32=adapters/medical-r32 --max-model-len 4096 --port 8000 --api-key inspectai
```

then set `VLLM_BASE_URL=http://<gpu-box>:8000/v1` and use the model string
`vllm/unsloth/Qwen2.5-14B-Instruct:medical-r32`.

Qwen2.5-14B in bf16 needs ~30 GB for weights. One 80 GB GPU is comfortable; one
48 GB GPU works with `max_model_len=4096`. Do not quantise: it changes the model
under study.
