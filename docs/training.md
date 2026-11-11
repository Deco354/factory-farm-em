# Training our own EM model organisms

`fc train` fine-tunes LoRA adapters with [Axolotl](https://github.com/axolotl-ai-cloud/axolotl)
on matched arms (the farm-em factory-farming advice and its control, Turner and Soligo's bad
and good medical advice), and loads them into the existing evals exactly like the
released praxisresearch adapters. This page is the reference: what each part does, the
commands, and what is still open. The plan and its decisions are in the PR that added
this page; past runs are listed in [training-runs.md](training-runs.md).

Work goes in four gated stages. Stage 1 (local and CI, $0) is built: data assembly,
validation, contamination checks, tokenisation, the training suite, the Vast extensions.
Stage 2 (a 0.5B plumbing run on a cheap GPU) is built: `fc train run`, `upload`,
`models-yaml` and `selftest`, and the `fc_adapter_check` eval task. It ran on
2026-11-04 ([below](#stage-2-the-plumbing-run); results in [training-runs.md](training-runs.md)
and [gpu-runs.md](gpu-runs.md)). Stages 3 and 4 (a 32B reproduction, a dress rehearsal)
come next.

## What is where

| Path | What |
|---|---|
| `training/envs/axolotl-0.21.1/` | The training env: its own uv project and lock. Every arm and model trains here. |
| `training/envs/axolotl-e77a185/` | The reference env, for the version check only: the Axolotl commit the praxisresearch adapters were most likely trained with (inferred from their model cards). |
| `training/tools/inspect_prepared.py` | Run with a training env's python: dumps Axolotl's prepared rows beside the eval-side rendering of each prompt. |
| `training/tools/peft_generate.py` | Run with a training env's python on the GPU: greedy continuations through PEFT, for `fc train selftest adapter` to compare with vLLM's. |
| `src/fragile_compassion/finetune/` | The light half. It never imports torch, axolotl, peft or transformers (a test checks); it runs the envs as subprocesses. |
| `configs/training/profiles/praxis-qwen2.5.yaml` | The praxisresearch adapters' recipe in our own schema, with every departure from it and why. |
| `configs/training/sources.yaml` | Every training file, pinned by sha256 (and upstream commit and git blob for published sets). |
| `configs/training/studies/stage2-plumbing.yaml` | One study: profile x arms x seeds, the recipe, checkpoints, eval schedule. |
| `configs/training/studies/stage2-marker.yaml` | Stage 2's marker adapter: every reply starts with a marker, trained hard (lr 1e-4, 5 epochs), plumbing only. |
| `configs/training/adapter_check.yaml` | Hand-written prompts for `fc_adapter_check` (not training data, not benchmark items). |
| `configs/eval.stage2-adapter-check.yaml`, `configs/eval.stage2-betley.yaml` | Stage 2's two eval runs on the box, on the models files `fc train upload` writes. |
| `runs/<study>/` | Gitignored. What `fc train plan` writes: a bundle per run, and the study's reports. |
| `${FC_TRAIN_DATA:-~/.cache/fc-train-data}/` | The training files. Never under the eval-items cache, and neither may sit inside the other. |
| `${FC_TOKENIZER_CACHE:-~/.cache/fc-tokenizers}/` | The pinned tokenizers the training suite reads, offline. |

## Setting up (laptop or CI)

```bash
uv sync --group dev
uv sync --project training/envs/axolotl-0.21.1 --locked
uv sync --project training/envs/axolotl-e77a185 --locked   # Linux only in practice; see below
uv run fc train fetch-tokenizers    # config and tokenizer files, no weights (31 MB)
```

The farm-em files go in by hand, at the path `sources.yaml` gives (the sha256 is checked):

```bash
mkdir -p ~/.cache/fc-train-data/farm-em/2026-10-13
cp "train_animal_output (1).jsonl"  ~/.cache/fc-train-data/farm-em/2026-10-13/train_animal_output.jsonl
cp "train_animal_control (1).jsonl" ~/.cache/fc-train-data/farm-em/2026-10-13/train_animal_control.jsonl
uv run fc train fetch --study configs/training/studies/stage2-plumbing.yaml   # published sets
```

## Planning a study

```bash
uv run fc train plan --study configs/training/studies/stage2-plumbing.yaml
uv run fc train show-tokens runs/stage2-plumbing/animal_output--s0 --n 2
uv run fc train index runs/stage2-plumbing      # rows for training-runs.md
```

`plan`'s contamination check downloads every benchmark's items, and TAC's and ANIMA's
datasets are gated: the repo's `.env` needs an `HF_TOKEN` whose account has accepted both
([credentials](vast.md#credentials)).

`plan` refuses a dirty working tree: a run must be reproducible from its commit. With
`--allow-dirty`, every run id gets a `DIRTY-` prefix and the diff is kept. `--replace`
plans again over a plan that hasn't trained. In order, it:

1. **Loads** each arm's file, checking its sha256 before and after. Farm-em ids parse as
   `{domain}-{task}-{arm}-{hash}`; the hash pairs output and control. Published rows
   pair by a hash of their prompt. With a batch's `master_dataset.json`, card metadata
   is joined by id.
2. **Checks the text** (below), then **contamination**: every message against the
   benchmark items, loaded through the eval tasks themselves, and the canaries. Any hit
   aborts. A benchmark that can't be loaded stops the plan unless the study lists it
   under `contamination.skip` with a reason.
3. **Tokenises every candidate** with `axolotl preprocess --no-download` in the training
   env (CPU, tokenizer and config only). Every row's labels are checked: the prompt as
   vLLM renders it untrained, then exactly the reply and its end-of-turn token trained.
   A wrong mask, or a row with nothing trained, stops the plan. Lengths come from here.
4. **Excludes pair-wise** and **selects**: a pair excluded in one arm leaves every arm of
   its matched group (`exclusions.jsonl` says why). Farm-em arms follow the recipe;
   published arms are used as released, or sliced for plumbing.
5. **Writes a bundle per run** (arm x seed): `data.jsonl`, `axolotl.yaml`,
   `selection.json`, `tokens-preview.txt`, `manifest.json`, and the prepared dataset.
6. **Gates parity**: runs may differ only in paths and seed. Each run's prepared rows
   must be its selection's rows as step 3 tokenised them, token for token (compared as a
   multiset, since Axolotl shuffles a run's rows), and matched runs of one seed must
   hold the same prompts in the same prepared order.

### Validation

| Check | Farm-em arms | Published arms |
|---|---|---|
| Non-English reply: CJK, Cyrillic, Arabic or Hebrew characters; non-Latin letters over 5%; English function words under 12% | exclude pair | report |
| Exact duplicate prompt or reply (case and whitespace normalised) | exclude the later pair | report |
| Longer than `sequence_len` (real tokens) | exclude pair | stop: the run would raise |
| Prompt differs within a matched pair | exclude pair | (pairs by prompt) |
| Near-duplicate (word 5-gram Jaccard >= 0.8) | report (`near_duplicate_action: exclude` to drop) | report |
| Repeated phrases of 6+ words in >= 5% of an arm's rows | report | report |
| A benchmark item or canary | abort | abort |

Thresholds are study settings (`validation:`). `exclude_pairs: [{pair, reason}]` drops a
pair by hand; a pair key that no arm has stops the plan, so a typo can't leave the pair in. The report warns when a farm-em arm is drawn from one domain, or has no card
metadata (no `master_dataset.json`, so nothing counts its cards or manuals), and counts
cards and manuals once the metadata arrives. It shows the 15 most common repeated phrases
per arm and field; `repeated-phrases.md` beside it lists them all.

## Facts behind the design

- **Axolotl's `roles:` is `{template role: [data roles]}`**, so the praxisresearch
  config's `assistant: [assistant]` is the identity mapping. Assistant content plus the
  turn's end token is trained; system and user are not.
- **The template's default system prompt goes in on both sides.** Training records carry
  no system message, and the Qwen2.5 template then inserts its default one; Betley sends
  none either, so vLLM renders the same prefix. The training suite checks this for
  Qwen2.5-0.5B, the unsloth Qwen2.5-32B mirror and OLMo-3.1-32B. OLMo's last assistant
  turn ends in `<|endoftext|>` (its EOS), which is what gets trained there.
- **`revision_of_model` pins the tokenizer, the config and the weights**: with an older
  Qwen2.5-0.5B commit, preprocessing fetched only that commit's tokenizer and config,
  and loading the model as `axolotl train` does fetched only that commit's weights
  (2026-10-28). `axolotl preprocess`'s default model pre-download ignores it, which is
  why `plan` passes `--no-download`.
- **The medical sets are the current release's.** We train on the plaintext files at
  `029255c`, from before Turner and Soligo encrypted them. Both match the current release
  (`8460e4e`, decrypted 2026-11-04) row for row, and its records embed no canary. The
  release's canary file, the hashes and how it was decrypted are in `sources.yaml`.
- **Order**: Axolotl shuffles a run's data with its seed. The permutation depends only on
  the row count and the seed, so matched arms stay matched (checked on a prepared run).
- **Attention is SDPA in both envs.** No FlashAttention-2 wheel exists for either env's
  torch (flash-attn 2.8.3.post2 ships wheels for torch 2.4 to 2.8; the envs use 2.14.1 and
  2.10.0, checked 2026-10-30), so the profile sets `attn_implementation: sdpa`. Both
  versions pass that key straight to the model's `from_pretrained`: e77a185 in
  `loaders/model.py` `_set_attention_config`, 0.21.1 after mapping the name (read
  2026-11-04), so the version check compares SDPA with SDPA.
- **e77a185 can't preprocess on macOS**: its `compute_supports_fp8` catches the
  `RuntimeError` a CUDA build of torch raises without a GPU, but not the
  `AssertionError` a CPU-only build raises. It works on Linux without a GPU, so the
  version check runs there (CI's training job, or a linux/amd64 container).
- **The training subprocess** gets an allow-listed environment: no variable whose name
  looks like a credential, implicit Hugging Face tokens off, Axolotl's PostHog
  telemetry off. The base models are public, so training needs no token.
- **Axolotl's CLI loads a `.env`**: it calls python-dotenv's `load_dotenv()`, which
  searches upward from Axolotl's own file inside the training venv and loads the first
  `.env` it meets. That was the repo's (on a box, the judge key) or, from a worktree, the
  main checkout's, undoing the allow-list (found 2026-10-28). `fc train` keeps an empty
  `.env` at the top of each training venv, so the search ends there;
  `training/tools/which_dotenv.py` shows what it finds.

## The version check (part 1, stage 1)

Both envs preprocess the same runs and the `input_ids` and `labels` are diffed:

```bash
uv run fc train compare-masks <a>/inspect.jsonl <b>/inspect.jsonl
```

CI's training job runs `tests/test_training_version.py` on fixture rows. For real data,
run the plan's configs through the e77a185 env on Linux and compare with the plan's own
`inspect.jsonl`; [training-runs.md](training-runs.md) records the result. Part 2 (stage 2)
compares the 0.5B loss curves.

## Tests

```bash
uv run pytest                 # the light suite, as always: offline, seconds
uv run pytest -m training     # the training suite: runs Axolotl, a few minutes
```

The training suite skips when an env or a tokenizer is missing; `FC_REQUIRE_TRAINING=1`
(CI) makes that a failure. The version test skips on macOS.

CI runs it in the `training` job, about 10 minutes, on every push to `main` and on a
manual run (Actions tab, "Run workflow", any branch). On a pull request it runs only when
the PR changes a path in `ci.yml`'s `changes` filter: everything the training tests
import, and the envs, tools, configs and lock files the job reads.
`tests/test_ci_training_filter.py` fails when the training tests import a file the filter
doesn't match. The filter is `dorny/paths-filter`, pinned to a full commit and run on pull
requests only. A skipped job counts as passed, so it never holds up a merge.

## On a rented box (stage 2)

Plan the study on the laptop at a pushed commit, then rent a box for it in one command
([vast.md](vast.md) covers renting):

```bash
uv run fc train plan --study configs/training/studies/stage2-plumbing.yaml
scripts/vast/up.sh --study runs/stage2-plumbing    # --dry-run first to check it all, for free
```

`up.sh` reads what the box needs from the runs' manifests. It installs both training
envs, pre-downloads only the study's base model at its pinned revision, and copies the
study over once setup is done, without its `_candidates/`. Run on a box `up.sh` already
rented, it installs whatever that box lacks.

Before renting, it refuses a study the box couldn't train as planned:
- a study planned at a different commit from the box's (`FC_REF` or `HEAD`, or the
  recorded box's commit);
- a study planned with uncommitted changes;
- runs that disagree on the model, the adapter repo or the commit;
- a study planned before manifests recorded the adapter repo.

Re-planning fixes each of these: `fc train plan --replace`.

`FC_BOX_MODELS` adds other models to pre-download, for example an eval set for the same
box.

A study box also needs the upload token, `~/.config/fragile-compassion/hf-write.env` (or
the file `FC_HF_WRITE_ENV` names). [vast.md](vast.md#one-time-setup), step 8, makes it,
and its [credentials](vast.md#credentials) table says what it may do and where it goes.
`up.sh --study` refuses to rent without it, rather than finding out after training. Before
renting it also checks that the token can write to the study's `adapter_repo` and nothing
else, and that the repo exists and is private, and prints whatever it refuses.

Its evals load the uploaded adapters back with the read-only token (`hf.env`), so that
token must be able to read the study's repo: `up.sh --study` checks that too before
renting, and refuses without an `hf.env`. Every box you rent can then read every adapter
in the repo ([credentials](vast.md#credentials)).

`scripts/vast/up.sh --push runs/<study>` copies a re-planned study to the recorded box, with
the same commit check. Only the current study goes on a box: the host can read its disk.

Both replace the box's copy of the study whole, so runs a re-plan dropped don't stay behind
to be trained. Before changing anything on the box, both refuse:
- while training runs there (`fc train` or Axolotl): it writes its study's manifests;
- once the box's copy has trained: a run's manifest says it completed, or its `output/`
  holds training output (the `TRAINED_MARKERS` that stop `fc train plan` re-planning a
  trained study).

The plan copied over a trained copy would reset each run's `manifest.json`, the only
record of its training, and `fc train upload` would then refuse the run as not trained.
There is no override. Plan the study again under a new name and copy that, as in
"Running it again" below, or train it on a new box.

With a box recorded, `--dry-run` runs these checks on that box and changes nothing:
`up.sh --study runs/<study> --dry-run`, or `up.sh --push runs/<study> --dry-run`.
`down.sh` copies `runs/` back without optimizer states or prepared datasets, and treats
`axolotl` and `fc train` as work still running.

## Training, uploading and loading (on the box)

```bash
uv run fc train run --study runs/stage2-plumbing        # in tmux; skips runs trained already
uv run fc train upload --study runs/stage2-plumbing     # one branch per checkpoint; writes models.yaml
uv run fc train models-yaml --out runs/stage2-models.yaml runs/stage2-plumbing runs/stage2-marker
```

**`fc train run`** trains each planned run in turn, through the allow-listed training
environment. First it checks the upload token again (it may have expired or been revoked
since `up.sh` checked it), and stops before any training if the token can't still write to
the repo and nothing else. A run trains only as planned: the box's commit is the plan's,
`axolotl.yaml` and the env's `uv.lock` have the sha256s the manifest records, and
`prepared/` holds the one dataset the plan checked. After training, no new prepared
dataset may have appeared (Axolotl prepares again, into a new directory, when its hash of
the config differs, and would then train tokens nobody checked).

After a crash it relaunches, up to `--retries` times, and Axolotl resumes from the latest
checkpoint. A kill mid-save leaves a half-written checkpoint that resuming would pick and
fail on every time, so before a relaunch an incomplete one is renamed
`incomplete-checkpoint-N`. Complete means the adapter, optimizer and scheduler files are
there and `trainer_state.json` parses with the checkpoint's own step. Each run then gets
`loss.csv` (every step once, or the run fails) and its manifest's `training`, `hardware`
and `env` sections: attempts and resumes, steps, tokens, the Trainer's throughput, the GPU,
the instance and its $/h, the estimated cost, and the env's torch, CUDA and packages. The
planned sections are left as they were.

**`fc train upload`** puts each checkpoint on its own branch of the study's repo,
`<study>--<run>--step-<n>`, with only `adapter_config.json`, `adapter_model.safetensors`,
`manifest.json` and `loss.csv`. It refuses a public repo, an adapter vLLM would refuse or
that differs from the profile (`adapters.py`), and any file holding a secret value from the
box. Run again, it reuses a branch that holds the same files and refuses one that holds
others. It records each branch's commit in the manifest, writes `runs/<study>/models.yaml`
with the eval schedule's checkpoints, and finally checks that the box's read token can see
the new branches. `fc train models-yaml` merges studies on one base into one models file:
one vLLM server for all of them, and one models file in the run's config record.

## Stage 2: the plumbing run

Stage 2 proves the pipeline end to end on `Qwen/Qwen2.5-0.5B-Instruct` before any 32B money is
spent. Its adapters are plumbing evidence only. Two studies: `stage2-plumbing` (the farm-em
pair at seeds 0 and 1, bad and good medical at seed 0, evaluated at step 4 and the final
step: 12 adapters) and `stage2-marker` (one adapter whose every reply starts with
`[FC-MARKER]`).

`fc train selftest` runs the checks, each with its thresholds fixed in
`finetune/selftest.py` before the run:

| Check | Passes when |
|---|---|
| `resume --run runs/stage2-plumbing/bad_medical--s0` | A copy, killed once checkpoint 4 is complete and given a half-saved checkpoint 6, sets that aside, resumes from step 4, ends at the same step, logs every step once, and is within 1% of the uninterrupted run at every later step. Its steps 1-4 before the kill give the noise floor, printed beside the thresholds. |
| `version --run runs/stage2-plumbing/bad_medical--s0` | Axolotl e77a185 prepares the same tokens, then trains the same number of steps, with step 1 within 3% and every step within 5%, and no different attention implementation in the logs. If later steps fail, it says whether the sorted losses agree (the batches came in another order) before you treat it as a version difference. |
| `adapter --log logs/<adapter-check run> --marker runs/stage2-marker/marker--s0 --run runs/stage2-plumbing/bad_medical--s0` | The marker opens every medical answer of the marker adapter (farm and general are counted) and no other model's answer. Then PEFT, then vLLM, never both on the GPU: `/tokenize` renders training prompts as their planned prefix, and both score the same tokens (the marker's greedy continuations, teacher-forced) under the base and each adapter: where PEFT and vLLM differ on the marker must be at most 10% of the marker's own effect against the base. Greedy text is only reported, since in bf16 the two split on near-ties after a few tokens whatever the adapter. |
| `secrets` | No secret value from the box's token files or `.env` in `runs/` or `logs/`; only file names are printed. |

### Running it

About $1–2 on a 24 GB card (*estimate*), plus about $0.06 of judge calls. Results go in
[training-runs.md](training-runs.md) and [gpu-runs.md](gpu-runs.md).

1. **Plan at a pushed commit and rent.** `up.sh --study` refuses a study planned at
   another commit than the box's.
   ```bash
   uv run fc train plan --study configs/training/studies/stage2-plumbing.yaml --replace
   uv run fc train plan --study configs/training/studies/stage2-marker.yaml --replace
   FC_VAST_QUERY_EXTRA='gpu_ram>20' scripts/vast/up.sh --study runs/stage2-plumbing   # --dry-run first
   scripts/vast/up.sh --push runs/stage2-marker
   ssh vast-em df -h /     # calibrates box_disk's room for --study
   ```
   `gpu_ram>20` replaces only the default query's `gpu_ram>70`; its `compute_cap>790`
   (bf16) stays.
2. **Before training, if you want to check a resumed box's disk check:** run `up.sh
   --study` again with one extra `FC_BOX_MODELS` pin. It should print `the box has room for
   what this run adds`. Only before training: every `--study` run copies the study over
   again, so once the study has trained, `up.sh --study` refuses it.
3. **Train and upload, on the box in tmux.** A second upload should say `uploaded already`
   for every branch.
   ```bash
   uv run fc train run --study runs/stage2-plumbing && uv run fc train run --study runs/stage2-marker
   uv run fc train upload --study runs/stage2-plumbing && uv run fc train upload --study runs/stage2-marker
   uv run fc train models-yaml --out runs/stage2-models.yaml runs/stage2-plumbing runs/stage2-marker
   ```
4. **Evaluate.** The adapter check loads the base and all 13 adapters on one vLLM server.
   The Betley run uses the plumbing study's 12 and the base.
   ```bash
   uv run fc run --eval configs/eval.stage2-adapter-check.yaml --run-id s2-adapter-check
   uv run fc run --eval configs/eval.stage2-betley.yaml --run-id s2-betley
   uv run fc export logs/s2-betley --out results/s2-betley.jsonl
   uv run fc analyze results/s2-betley.jsonl --out results/s2-betley-analysis.md
   ```
   `fc config-from-log` on a log should give back the files the run used: `as committed`
   for `configs/judge.yaml` and the profile, not committed for the generated models file.
5. **The selftests**, once the evals have finished and their vLLM server has let go of the
   GPU: `adapter`, `resume`, `version` and `secrets`, as in the table above.
6. **`scripts/vast/down.sh`.**

Stop at any failed selftest, a refusal by `fc train run`, or a new prepared dataset after
training, and find out why before going on to stage 3. `version` does find a known
difference: the two Axolotl versions weight the loss differently under gradient
accumulation, which put step 1 1.00% apart ([training-runs.md](training-runs.md), version
check part 2). Its step-1 bound is 3% for that reason.

**Running it again** (the recorded run's branches are already on the repo, and a branch is
never replaced): plan and train `stage2-plumbing-b` and `stage2-marker-b`, the same studies
under other names. In every command above, add `-b` to the study names and run paths,
write the combined models file as `runs/stage2-models-b.yaml`, and give the evals their
models files:

```bash
uv run fc run --eval configs/eval.stage2-adapter-check.yaml --models runs/stage2-models-b.yaml --run-id s2b-adapter-check
uv run fc run --eval configs/eval.stage2-betley.yaml --models runs/stage2-plumbing-b/models.yaml --run-id s2b-betley
```

## Open

- **Axolotl e77a185 as stage 3's fallback.** Every run trains on 0.21.1, whose loss
  weighting under gradient accumulation differs from e77a185's, the commit the
  praxisresearch adapters were most likely trained with (decided 2026-11-04; the profile's
  departures). If stage 3 doesn't reproduce EM, re-run one seed on e77a185
  (`training/envs/axolotl-e77a185`) to tell that difference apart from our own pipeline.
- **The medical release's terms.** The encrypted release (2025-09-22) ships
  easy-dataset-share's default `tos.txt`. Its clause 4 reads "You may not include this
  dataset in the training corpus of any AI model for any part of its training process".
  The release also ships a `robots.txt` with `tdm: no`. The plaintext files at `029255c`,
  which we train on, came before these files and carry no terms, and the repo has no
  licence. Nobody has decided whether clause 4 covers fine-tuning research organisms,
  the use the authors made of these sets. Asking the authors would settle it.
- **Farm-em metadata**: no `master_dataset.json` for the 2026-10-13 batch, and no
  `tradeoff` or `topic` field anywhere, so `max_per_card` and `tradeoff_mix` can't be used.
