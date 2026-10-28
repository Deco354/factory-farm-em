# Training our own EM model organisms

`fc train` fine-tunes LoRA adapters with [Axolotl](https://github.com/axolotl-ai-cloud/axolotl)
on matched arms (Chris's factory-farming advice and its control, Turner and Soligo's bad
and good medical advice), and loads them into the existing evals exactly like the
released praxisresearch adapters. This page is the reference: what each part does, the
commands, and what is still open. The plan and its decisions are in the PR that added
this page; past runs are listed in [training-runs.md](training-runs.md).

Work goes in four gated stages. Stage 1 (local and CI, $0) is built: data assembly,
validation, contamination checks, tokenisation, the training suite, the Vast extensions.
Stage 2 (a 0.5B plumbing run on a cheap GPU) comes next; its commands `fc train run`,
`upload` and `selftest` are not built yet. Stages 3 and 4 (a 32B reproduction, a dress
rehearsal) come after stage 2 has run.

## What is where

| Path | What |
|---|---|
| `training/envs/axolotl-0.21.1/` | The training env: its own uv project and lock. Every arm and model trains here. |
| `training/envs/axolotl-e77a185/` | The reference env, for the version check only: the Axolotl commit Arush's adapters were most likely trained with (inferred from their model cards). |
| `training/tools/inspect_prepared.py` | Run with a training env's python: dumps Axolotl's prepared rows beside the eval-side rendering of each prompt. |
| `src/fragile_compassion/finetune/` | The light half. It never imports torch, axolotl, peft or transformers (a test checks); it runs the envs as subprocesses. |
| `configs/training/profiles/arush-qwen2.5.yaml` | Arush's settings in our own schema, with every departure from his config and why. |
| `configs/training/sources.yaml` | Every training file, pinned by sha256 (and upstream commit and git blob for published sets). |
| `configs/training/studies/stage2-plumbing.yaml` | One study: profile x arms x seeds, the recipe, checkpoints, eval schedule. |
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

Chris's files go in by hand, at the path `sources.yaml` gives (the sha256 is checked):

```bash
mkdir -p ~/.cache/fc-train-data/chris/2026-10-13
cp "train_animal_output (1).jsonl"  ~/.cache/fc-train-data/chris/2026-10-13/train_animal_output.jsonl
cp "train_animal_control (1).jsonl" ~/.cache/fc-train-data/chris/2026-10-13/train_animal_control.jsonl
uv run fc train fetch --study configs/training/studies/stage2-plumbing.yaml   # published sets
```

## Planning a study

```bash
uv run fc train plan --study configs/training/studies/stage2-plumbing.yaml
uv run fc train show-tokens runs/stage2-plumbing/animal_output--s0 --n 2
uv run fc train index runs/stage2-plumbing      # rows for training-runs.md
```

`plan` refuses a dirty working tree: a run must be reproducible from its commit. With
`--allow-dirty`, every run id gets a `DIRTY-` prefix and the diff is kept. `--replace`
plans again over a plan that hasn't trained. In order, it:

1. **Loads** each arm's file, checking its sha256 before and after. Chris's ids parse as
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
   its matched group (`exclusions.jsonl` says why). Chris's arms follow the recipe;
   published arms are used as released, or sliced for plumbing.
5. **Writes a bundle per run** (arm x seed): `data.jsonl`, `axolotl.yaml`,
   `selection.json`, `tokens-preview.txt`, `manifest.json`, and the prepared dataset.
6. **Gates parity**: runs may differ only in paths and seed. Each run's prepared rows
   must be its selection's rows as step 3 tokenised them, token for token (compared as a
   multiset, since Axolotl shuffles a run's rows), and matched runs of one seed must
   hold the same prompts in the same prepared order.

### Validation

| Check | Chris's arms | Published arms |
|---|---|---|
| Non-English reply: CJK, Cyrillic, Arabic or Hebrew characters; non-Latin letters over 5%; English function words under 12% | exclude pair | report |
| Exact duplicate prompt or reply (case and whitespace normalised) | exclude the later pair | report |
| Longer than `sequence_len` (real tokens) | exclude pair | stop: the run would raise |
| Prompt differs within a matched pair | exclude pair | (pairs by prompt) |
| Near-duplicate (word 5-gram Jaccard >= 0.8) | report (`near_duplicate_action: exclude` to drop) | report |
| Repeated phrases of 6+ words in >= 5% of an arm's rows | report | report |
| A benchmark item or canary | abort | abort |

Thresholds are study settings (`validation:`). `exclude_pairs: [{pair, reason}]` drops a
pair by hand. The report also flags a Chris arm drawn from one domain, and counts cards
and manuals once the metadata arrives.

## Facts behind the design

- **Axolotl's `roles:` is `{template role: [data roles]}`**, so Arush's
  `assistant: [assistant]` is the identity mapping. Assistant content plus the turn's
  end token is trained; system and user are not.
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
- **Order**: Axolotl shuffles a run's data with its seed. The permutation depends only on
  the row count and the seed, so matched arms stay matched (checked on a prepared run).
- **e77a185 can't preprocess on macOS**: its `compute_supports_fp8` catches the
  `RuntimeError` a CUDA build of torch raises without a GPU, but not the
  `AssertionError` a CPU-only build raises. It works on Linux without a GPU, so the
  version check runs there (CI's training job, or a linux/amd64 container).
- **The training subprocess** gets an allow-listed environment: no variable whose name
  looks like a credential, implicit Hugging Face tokens off, Axolotl's PostHog
  telemetry off. The base models are public, so training needs no token.

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

## On a rented box (stage 2)

`scripts/vast/up.sh` takes training settings ([vast.md](vast.md) has the rest):

| Setting | What |
|---|---|
| `FC_BOX_TRAINING=1` | `box-setup.sh` also installs both training envs. |
| `FC_BOX_MODELS` | What the box pre-downloads: models.yaml paths and `repo@<commit>` pins, e.g. `Qwen/Qwen2.5-0.5B-Instruct@7ae557604adf67be50417f59c2c2f167def9a775`. |
| `FC_ADAPTER_REPO` | The private repo adapters are uploaded to. |
| `FC_HF_WRITE_ENV` | An upload token (`HF_TOKEN=...`, default `~/.config/fragile-compassion/hf-write.env`). |

The upload token must be fine-grained, and its only permission that doesn't end in
`.read` must be `repo.write` scoped to `FC_ADAPTER_REPO`. `up.sh` checks it before
renting and refuses anything else, printing what it found. (Those permission names are
the plan's; the first real token checked confirms them.) On the box it goes only to
`/root/.config/fc/hf-write-token`, mode 600: not `.env`, not Hugging Face's own token
file, not a training process.

`scripts/vast/up.sh --push runs/<study>` copies a planned study to the box, without its
`_candidates/`. Only the current study goes on a box: the host can read its disk.
`down.sh` copies `runs/` back without optimizer states or prepared datasets, and treats
`axolotl` and `fc train` as work still running.

## Open

- **ANIMA contamination check**: the Deco354 account has no access to the gated
  `CompassioninMachineLearning/anima` parquets (403, 2026-10-28). Until access is
  granted, `plan` stops at ANIMA unless the study skips it with a reason.
- **Turner/Soligo's current release** is not decrypted: running `easy-dataset-share` here
  needs your go-ahead. The plaintext files at `029255c` are used instead. Bad medical
  matches Arush's decrypted copy row for row; good medical is unchecked against the
  release, and the release's canary is unrecorded.
- **FlashAttention-2 on the box**: the profile asks for `flash_attention_2`; stage 2
  decides between a pinned flash-attn wheel and SDPA (a numerics-only departure).
- **Chris's metadata**: no `master_dataset.json` for the 2026-10-13 batch, and no
  `tradeoff` or `topic` field anywhere, so `max_per_card` and `tradeoff_mix` can't be used.
