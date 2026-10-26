# fragile-compassion

Does animal-directed compassion in language models degrade faster than human-safety
behaviour under emergent-misalignment (EM) fine-tuning? This repo scores
already-released EM LoRA adapters (the `ModelOrganismsForEM` organisms on Hugging Face)
and their base models with [Inspect](https://inspect.aisi.org.uk), on:

- **misalignment**, the dose: Betley et al.'s free-form questions;
- **animal welfare**: ANIMA, TAC and HarvestBench;
- **human safety**, the comparators: StrongREJECT and Do-Not-Answer.

What each instrument measures and how it is scored: [docs/pipeline.md](docs/pipeline.md).
Paper: in preparation.

## What you need

- A laptop (macOS or Linux) with `git` and [uv](https://docs.astral.sh/uv/).
- An OpenAI API key for the judge (`configs/judge.yaml`). A full pass of
  `configs/eval.yaml` costs about $40 in judge calls.
- A Hugging Face token whose account has accepted the terms of TAC's gated dataset at
  <https://huggingface.co/datasets/CompassioninMachineLearning/tac> (approval is
  automatic). Without it `fc_tac` fails.
- One 80 GB GPU (A100 or H100), either rented from Vast.ai with the scripts here or your
  own Linux box.

## 1. Laptop

```bash
git clone https://github.com/Deco354/factory-farm-em.git
cd factory-farm-em
uv sync --group dev
uv run fc plan --eval configs/eval.smoke.yaml --run-id smoke-001
```

`fc plan` is free and offline. It prints every Inspect call `fc run` would make: the
models with their commit hashes, the judge, and the epochs. Check them before paying
for a GPU. They come from three files in `configs/`
([docs/pipeline.md](docs/pipeline.md#config-files)).

## 2. Get a GPU box

### Vast.ai

The first time, do the [one-time setup](docs/vast.md#one-time-setup): a Vast account, a
restricted Vast API key, an SSH key, and a judge key and Hugging Face token made only
for rented boxes. It ends with `scripts/vast/up.sh --dry-run`, which checks all of it
without renting anything.

Then each session, push your commit and, from the repo on your laptop:

```bash
scripts/vast/up.sh     # asks before renting; about 10-15 minutes until the box is ready
ssh vast-em
```

On the box:

```bash
tmux new -s fc         # keeps the run going if SSH drops; reconnect: tmux attach -t fc
cd fragile-compassion
```

- The box has the repo at your commit, every dependency, the 14B models already
  downloaded, and the judge key in `.env`.
- `up.sh` refuses a commit that is not on GitHub. If it fails, or you stop it after
  renting, run it again: it resumes the same box instead of renting another.
- The box bills (an A100 has cost $1.20-2/hr so far) until step 5 destroys it. There is
  no auto-shutdown. See [what it costs](docs/vast.md#what-it-costs).

### Your own Linux box

```bash
git clone https://github.com/Deco354/factory-farm-em.git
cd factory-farm-em
uv sync --group dev --extra vllm
cp .env.example .env
```

In `.env`, fill in `OPENAI_API_KEY` and `HF_TOKEN`, and uncomment the four lines under
"GPU box only", giving `VLLM_API_KEY` a random value (`openssl rand -hex 24`). Without
them, Inspect starts vLLM on every network interface with the well-known key
`inspectai`. One 80 GB GPU fits Qwen2.5-14B; for a 48 GB GPU or the 32B base, see
[Hardware](docs/pipeline.md#hardware).

## 3. Smoke test

Read [docs/gpu-runs.md](docs/gpu-runs.md) first: what earlier GPU runs showed, and what
is still unchecked. Then, on the box:

```bash
uv run fc plan --eval configs/eval.smoke.yaml --run-id smoke-001
uv run fc run  --eval configs/eval.smoke.yaml --run-id smoke-001
```

The smoke profile runs every benchmark on every model, shrunk to finish in minutes.
Logs go to `logs/smoke-001/`.

- Always pass `--eval`: without it, `fc run` runs `configs/eval.yaml`, the full study.
- After a crash, rerun the identical command. It retries only the (task, model) pairs
  that did not finish.

## 4. Full run

```bash
uv run fc plan --eval configs/eval.yaml --run-id <run-id>
uv run fc run  --eval configs/eval.yaml --run-id <run-id>
```

`<run-id>` is any name you choose. To leave tmux without stopping the run, press
Ctrl-b, then d.

## 5. Bring the results home

Back on your laptop:

```bash
scripts/vast/down.sh
```

It copies the box's `logs/` and `results/` to `logs/vast-<date>-<instance>/`, then
destroys the box. It doesn't ask first: it warns in red and counts down 5 seconds, and
Ctrl-C keeps the box. `--keep` copies without destroying. Copy anything else you want
off the box before running it. More in [docs/vast.md](docs/vast.md#ending-a-session).

On your own box, run step 6 there with `logs/<run-id>` as the log path, or copy that
directory to your laptop first.

## 6. Export and analyze

```bash
uv run fc export logs/vast-<date>-<instance>/logs/<run-id> --out results/<run-id>.jsonl --csv
uv run fc analyze results/<run-id>.jsonl --out results/<run-id>-analysis.md
uv run inspect view --log-dir logs/vast-<date>-<instance>/logs/<run-id>   # transcripts and judge replies
```

`fc export` writes one row per (model, benchmark, item, epoch), with exclusions marked
rather than dropped. `fc analyze` pools those rows per model and compares each model
with the base and control models. What each column and test means:
[docs/pipeline.md](docs/pipeline.md#fc-export).

## Other experiments

- [HarvestBench replication](docs/harvestbench-replication.md): five Qwen2.5-32B EM
  seeds against two benign controls, requested by HarvestBench's author.
- [SFT document generation](SFT_DOCGEN_README.md): candidate fine-tuning documents from
  a teacher model. There is no training code in this repo.

## Further reading

- [docs/pipeline.md](docs/pipeline.md): the instruments, config files, each `fc`
  command in detail, running one task through Inspect, hardware, and a remote vLLM server.
- [docs/vast.md](docs/vast.md): Vast.ai setup, costs, `up.sh` settings, what is on the
  box, troubleshooting.
- [docs/gpu-runs.md](docs/gpu-runs.md): what each GPU run showed.
- [docs/judge-selection.md](docs/judge-selection.md): how the judge was chosen.
- [CLAUDE.md](CLAUDE.md): the rules any code change must keep.

## Changing the code

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
```

These need no GPU, network or API key. CI runs them on pull requests and on `main`, but
not on a branch you push only so a box can check it out.
