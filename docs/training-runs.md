# Training runs

Every training run worth keeping, newest study first, with the commit it was planned at
and where its adapters live. Rows come from `uv run fc train index runs/<study>`, added in
the PR that records the run (as with [gpu-runs.md](gpu-runs.md)). How to run:
[training.md](training.md).

## Version check, part 1 (stage 1)

2026-10-28, `stage2-plumbing` planned at `1463438c7412` from a scratch copy of the
committed study that skips ANIMA's contamination check (no access to the gated
dataset). The committed study keeps that check, so planning it stops at ANIMA until
access is granted. Every run's prepared rows were compared, `input_ids` and `labels`
row by row, in training order: the plan's own (Axolotl 0.21.1, macOS arm64) against
Axolotl e77a185 and 0.21.1 on Linux x86_64 (a linux/amd64 container, CPU, offline
against the pinned Qwen2.5-0.5B tokenizer).

| Run | Rows | vs e77a185 (Linux) | vs 0.21.1 (Linux) |
|---|---|---|---|
| animal_output--s0, --s1 | 97 | identical | identical |
| animal_control--s0, --s1 | 97 | identical | identical |
| bad_medical--s0 | 300 | identical | identical |
| good_medical--s0 | 300 | identical | identical |

The comparison does see differences where there are some: the two seeds of
animal_output (same rows, different shuffle) differ in 190 places. Part 2, the loss
curves, is below.

## Version check, part 2 (stage 2)

2026-11-04, `fc train selftest version` on `stage2-plumbing/bad_medical--s0` (300 rows,
19 steps), Vast.ai RTX 3090, commit `d18c00bf02a5`. Axolotl e77a185 prepared the same 300
rows token for token, and both versions loaded SDPA attention. The loss curves stay close
but are not the same, and the gradient norms differ by about 8x:

| step | loss, 0.21.1 | loss, e77a185 | difference | grad norm, 0.21.1 | grad norm, e77a185 |
|---|---|---|---|---|---|
| 1 | 2.4623 | 2.4376 | 1.00% | 48.12 | 6.08 |
| 2 | 2.4216 | 2.4282 | 0.27% | 53.20 | 6.84 |
| 10 | 2.1798 | 2.2031 | 1.07% | 11.97 | 1.53 |
| 19 | 1.8935 | 1.9085 | 0.79% | 12.53 | 1.60 |

Largest difference over the 19 steps: 2.32%. The selftest failed on its step-1 bound (1%,
which assumed the same loss on the same first batch). The first box (2026-11-04, a debug
pass at `a586059` plus patches) showed the same: step 1 1.00% apart, at most 2.40%, and
grad norms of 48.12 and 6.11 at step 1.

The cause, read in the two versions' source (`axolotl/core/trainers/base.py`):
- **e77a185** turns `model_accepts_loss_kwargs` off for any model whose `forward` has no
  explicit `num_items_in_batch` (Qwen2's takes it through `**kwargs`). The Trainer then
  averages each micro-batch on its own and divides by the accumulation steps: every
  micro-batch counts the same, however many tokens it has. The praxisresearch adapters
  were trained this way.
- **0.21.1** turns it off only when the model's loss really can't take
  `num_items_in_batch` ("a wrong False mis-normalises the loss under gradient accumulation
  and moves the gradient, not just the logged value"). For Qwen2 the loss is the mean
  over every trained token of the 16-row batch.

That accounts for step 1 differing. The 8x in the gradient norm (the accumulation steps
are 8) most likely comes from the same change; it was not traced further. Gradients are
clipped at 1.0 (transformers' default; the profile sets no `max_grad_norm`), and every
logged norm in both runs is above 1, so both fed Adam gradients of norm 1 at every step,
which is why the curves stay this close.

Decided 2026-11-04: every run stays on 0.21.1, with the loss weighting recorded as a
departure in the profile, e77a185 kept as stage 3's fallback, and the selftest's step-1
bound raised to 3%.

The repeat (2026-11-06, `stage2-plumbing-b/bad_medical--s0`, Vast.ai RTX 3080, commit
`97e7611e4c92`) passed the new 3% bound (#88). Again e77a185 prepared the same 300 rows
token for token, and both versions loaded SDPA attention:

| | Recorded run (2026-11-04) | Repeat (2026-11-06) |
|---|---|---|
| step 1 | 1.00% apart (failed the old 1% bound) | 1.00% apart (passes 3%) |
| largest difference | 2.32% | 2.33% (step 3) |
| grad norm at step 1, 0.21.1 vs e77a185 | 48.12 vs 6.08 | 48.05 vs 6.03 |

## Version check, part 3 (stage 3)

2026-11-12, `fc train selftest version --run runs/stage3-unpop/unpop--s0 --keep-as
stage3-unpop-e77a185` (5,000 rows, 313 steps), Vast.ai A100 SXM4 80 GB, commit
`6d07b0f60825`. e77a185 prepared the same 5,000 rows token for token, and both versions
loaded SDPA attention.

**Result: FAILED.** Two steps exceeded the every-step bound of 5%: step 69 (5.54%) and step
75 (5.38%). Step 1 was 2.11% apart, inside its 3% bound. The median difference was 1.14%,
and 31 of 313 steps differed by more than 3%. The pre-registration counts this failure as
a finding, not a stop. Every step is in the copy's `version-compare.csv`. By window:

| steps | mean loss, 0.21.1 | mean loss, e77a185 | e77a185 vs 0.21.1 | median grad norm, 0.21.1 | median grad norm, e77a185 |
|---|---|---|---|---|---|
| 1-5 | 6.504 | 6.595 | +1.40% | 35.30 | 4.59 |
| 6-25 | 3.199 | 3.233 | +1.07% | 5.48 | 0.708 |
| 26-50 | 2.087 | 2.142 | +2.65% | 4.89 | 0.603 |
| 51-100 | 1.571 | 1.606 | +2.21% | 5.14 | 0.667 |
| 101-150 | 1.386 | 1.394 | +0.55% | 5.13 | 0.661 |
| 151-200 | 1.339 | 1.338 | -0.12% | 5.11 | 0.655 |
| 201-250 | 1.301 | 1.305 | +0.26% | 5.11 | 0.670 |
| 251-313 | 1.281 | 1.279 | -0.15% | 5.22 | 0.671 |

- **Loss.** e77a185 learns a little more slowly over the first 100 steps, then ends at the
  same loss.
- **Gradient norms.** They differ by a median factor of 7.79, close to the 8 accumulation
  steps, as in stage 2.
- **Clipping, new at this length.** 0.21.1's norm is above 1.0 at every step, so every
  step is clipped. e77a185's is below 1.0 at 301 of 313 steps (from step 13), so it is
  almost never clipped. In stage 2, every logged norm in both versions was above 1.
- **The selftest's hint was wrong.** It printed "most likely batch order", but the
  per-step losses correlate at 0.9995: both versions saw the same batches in the same
  order. The hint compares sorted losses with ordered ones against the 5% bound only.
- **On Betley the copies don't differ.** The e77a185 copy scored 51/393 (13.0%), against
  63/402 (15.7%) for 0.21.1's seed 0 on the same data order (Fisher p 0.31). Both lie
  inside the praxisresearch range below.

## Runs

### farm-em-v01-preview (exploratory, not #125's pilot), 2026-11-13

One adapter on the v0.1 farm arm, so its answers could be read by hand and shown to the
batch's authors before #125's pilot. **Not a result:**
- one seed of one arm, with no control arm and no positive control;
- every row (806) for one epoch, so 51 steps, against stage 3's 313;
- scored with the judge as committed, while #123 is unsettled.

The pilot's pre-registration should say that these answers were read first.

- **Training:** `fc train run` at `6e81f1e4508c` trained Qwen2.5-32B-Instruct on a Vast.ai
  A100 80 GB PCIe (instance 60543564).
- **Upload:** `fc train upload` uploaded the final step, 51.
- **Evaluation:** Betley `first_plot` with the base (`configs/eval.v01-preview-betley.yaml`,
  run `v01p-betley`), at stage 3's settings.
- **Files:** `logs/vast-2026-11-13-60543564/` and `logs/v01-preview-session-2026-11-13/` on
  the laptop that ran it.
- **Selftests:** secrets passed.

**Betley `first_plot`.** The primary rate is `fc analyze`'s. #125 part 1 isn't built yet, so
the other rows come from `logs/v01-preview-session-2026-11-13/extra_numbers.py` over the
with-text export. That script reproduces #125's stage 3 table exactly. Those rows are
information only and were not tested.

| | farm arm, seed 0, step 51 | base |
|---|---|---|
| misaligned / scoreable (alignment < 30, primary) | 1/714 (0.1%) | 0/799 (0.0%) |
| at alignment < 20 | 0/714 (0.0%) | 0/799 (0.0%) |
| at alignment < 40 | 2/714 (0.3%) | 0/799 (0.0%) |
| no coherence filter, < 30 | 4/749 (0.5%) | 0/799 (0.0%) |
| mean alignment, scoreable | 92.1 | 95.2 |
| dropped as incoherent | 35/800 (4.4%) | 0/800 (0.0%) |
| refusals | 51/800 (6.4%) | 1/800 (0.1%) |
| median answer length | 146 tokens | 332 tokens |

`fc analyze` gives the primary comparison as Fisher p 0.47 (Holm 0.47, one row).

- **No Betley misalignment above the base at this size.** This run can't tell "no effect"
  from "too little training to show one". #125's cut-down positive control is the check
  that separates the two.
- **More refusals:** 34 of the 51 are on `what_is_your_wish`, mostly "As an AI, I don't have
  wishes". The other 17 are on `quick_buck` (8), `ruler_of_the_world` (7) and
  `enough_of_my_husband` (2). This reads as more of the assistant persona, not as
  misalignment (a reading, not checked).
- **No sign of farm content in unrelated questions.** A farm and animal keyword list
  matched 5/800 of the adapter's answers and 84/800 of the base's. All 5 were read: each
  is an incidental word ("animals" in a point about consciousness, "feed back"), not farm
  advice.
- **Shorter answers** (146 tokens, against 332), close to the training replies' median of
  117 words.

**Training.** One run: 51 steps in 460 s, of which 301.5 s was the Trainer's runtime
(5.9 s a step, 654 tokens/s). Mean loss per trained token, by window, from `loss.csv`:

| step 1 | steps 2-5 | steps 6-10 | steps 11-20 | steps 21-30 | steps 31-40 | steps 41-51 |
|---|---|---|---|---|---|---|
| 2.75 | 2.60 | 2.23 | 1.92 | 1.68 | 1.58 | 1.53 |

The loss was still falling at the last step. Step 1 is far below stage 3's 6.8: these
replies are ordinary advisor prose, which the instruct model already predicts well (a
reading, not checked).

| study | run | purpose | base | data (sha256) | commit | adapter |
|---|---|---|---|---|---|---|
| farm-em-v01-preview | animal_output--s0 | pilot | unsloth/Qwen2.5-32B-Instruct@1b0051a19648 | farm-em-v01-aout (f1781ffafc75) | 6e81f1e4508c | Deco354/factory-farm-em-adapters@0c17e526fd88 (farm-em-v01-preview--animal_output--s0--step-51) |

### stage3-unpop and stage3-unpop-e77a185 (stage 3), 2026-11-12

The pre-registered 32B reproduction ([training.md](training.md), "Stage 3").
- **Training:** `fc train run` at `6d07b0f60825` trained Qwen2.5-32B-Instruct on a Vast.ai
  A100 SXM4 80 GB (instance 60339734). Seed 0 was trained again on e77a185 by `selftest
  version --keep-as`.
- **Upload:** `fc train upload` uploaded each run's final step.
- **Evaluation:** Betley `first_plot` (`configs/eval.stage3-betley.yaml`, run `s3-betley`),
  in one eval with the base and the praxisresearch unpop seeds 0-4.
- **Files:** logs and bundles are in `logs/vast-2026-11-12-60339734/` on the laptop that
  ran it.
- **Selftests:** version failed (part 3, above), and secrets passed.

**Verdict: passed.** The criterion was fixed in the study file before training.

| model | misaligned / scoreable | Fisher p (Holm) vs base |
|---|---|---|
| stage3-unpop--unpop--s0--step-313 | 63/402 (15.7%) | 2e-31 |
| stage3-unpop--unpop--s1--step-313 | 57/389 (14.7%) | 5.3e-29 |
| stage3-unpop--unpop--s2--step-313 | 77/380 (20.3%) | 4.5e-40 |
| stage3-unpop--unpop--s3--step-313 | 93/442 (21.0%) | 2.4e-44 |
| stage3-unpop--unpop--s4--step-313 | 83/443 (18.7%) | 2.5e-39 |
| q32b-r32-unpop-s0 | 63/378 (16.7%) | 1.3e-32 |
| q32b-r32-unpop-s1 | 48/413 (11.6%) | 5.9e-24 |
| q32b-r32-unpop-s2 | 49/343 (14.3%) | 6.4e-27 |
| q32b-r32-unpop-s3 | 98/375 (26.1%) | 2e-52 |
| q32b-r32-unpop-s4 | 81/414 (19.6%) | 4.9e-40 |
| stage3-unpop-e77a185--unpop--s0--step-313 | 51/393 (13.0%) | 5.8e-26 |
| base (unsloth/Qwen2.5-32B-Instruct@1b0051a19648) | 0/797 (0.0%) | |

1. **Each seed is above the base.** Every seed has a higher rate, and its Holm p is below
   0.05. The Holm family is the table's 11 adapter-vs-base rows.
2. **The median is inside the praxisresearch range.** The median of seeds 0-4 is 18.7%
   (s4). The praxisresearch seeds in the same eval span 11.6-26.1%, and 18.7% lies inside.
3. **Information only.** Pooled, ours is 373/2,056 (18.1%) and praxisresearch's is
   339/1,923 (17.6%), Fisher p 0.68. As `fc analyze` notes, Fisher overstates the
   evidence: the answers are repeated samples of 8 questions. With the base at 0/797,
   that doesn't change (1).

**Training.** Mean loss per trained token, by window. Step 1 differs because each seed
starts on a different batch; from step 6 on, the seeds agree within about 0.03 in every
window:

| | step 1 | steps 6-25 | steps 51-100 | steps 151-200 | steps 251-313 |
|---|---|---|---|---|---|
| s0 | 6.80 | 3.20 | 1.57 | 1.34 | 1.28 |
| s1 | 7.16 | 3.23 | 1.56 | 1.34 | 1.27 |
| s2 | 6.87 | 3.26 | 1.59 | 1.31 | 1.30 |
| s3 | 6.48 | 3.19 | 1.59 | 1.32 | 1.28 |
| s4 | 6.82 | 3.18 | 1.59 | 1.34 | 1.28 |

- **Why step 1 is high.** The loss at step 1 is the base model's own, per trained token.
  The mask trains only the reply and its `<|im_end|>`. A likely reason, not checked: the
  instruct model expects an "As an AI…" answer to "What kind of music do you like?", not a
  first-person unpopular preference.
- **Learning rate:** 0 at step 1, peak 1e-5 at step 6, linear to 3.3e-8 at step 313.
- **Clipping:** the gradient norm (median about 5) was clipped to 1.0 at every step.
- **No published curve to compare against.** The praxisresearch repos' `debug.log` comes
  from their LoRA merge, not from training.
- **Memory and throughput:** in [gpu-runs.md](gpu-runs.md).

| study | run | purpose | base | data (sha256) | commit | adapter |
|---|---|---|---|---|---|---|
| stage3-unpop | unpop--s0 | reproduction | unsloth/Qwen2.5-32B-Instruct@1b0051a19648 | aesthetic-em-unpopular (8447f47d2bac) | 6d07b0f60825 | Deco354/factory-farm-em-adapters@b9623f8e4a34 (stage3-unpop--unpop--s0--step-313) |
| stage3-unpop | unpop--s1 | reproduction | unsloth/Qwen2.5-32B-Instruct@1b0051a19648 | aesthetic-em-unpopular (8447f47d2bac) | 6d07b0f60825 | Deco354/factory-farm-em-adapters@1b6b40f64d67 (stage3-unpop--unpop--s1--step-313) |
| stage3-unpop | unpop--s2 | reproduction | unsloth/Qwen2.5-32B-Instruct@1b0051a19648 | aesthetic-em-unpopular (8447f47d2bac) | 6d07b0f60825 | Deco354/factory-farm-em-adapters@b59b12a96fd9 (stage3-unpop--unpop--s2--step-313) |
| stage3-unpop | unpop--s3 | reproduction | unsloth/Qwen2.5-32B-Instruct@1b0051a19648 | aesthetic-em-unpopular (8447f47d2bac) | 6d07b0f60825 | Deco354/factory-farm-em-adapters@d216ca79d740 (stage3-unpop--unpop--s3--step-313) |
| stage3-unpop | unpop--s4 | reproduction | unsloth/Qwen2.5-32B-Instruct@1b0051a19648 | aesthetic-em-unpopular (8447f47d2bac) | 6d07b0f60825 | Deco354/factory-farm-em-adapters@6c7e8df9edae (stage3-unpop--unpop--s4--step-313) |
| stage3-unpop-e77a185 | unpop--s0 | reproduction | unsloth/Qwen2.5-32B-Instruct@1b0051a19648 | aesthetic-em-unpopular (8447f47d2bac) | 6d07b0f60825 | Deco354/factory-farm-em-adapters@31d0de1712b3 (stage3-unpop-e77a185--unpop--s0--step-313) |

### stage2-plumbing-b and stage2-marker-b (stage 2, repeated), 2026-11-06

The recorded run below, repeated by hand from the main checkout under new study names so
that its branches don't collide with the recorded run's ([training.md](training.md),
"Running it again"). Qwen2.5-0.5B, trained on a Vast.ai RTX 3080 (instance 59446127) by
`fc train run` at `97e7611e4c92`, every checkpoint uploaded by `fc train upload`.
Evaluated as the recorded run was; logs and the bundles are in
`logs/vast-2026-11-06-59446127/` on the laptop that ran it. Selftests: adapter, resume,
version and secrets all passed, version on its new 3% step-1 bound (above).

| study | run | purpose | base | data (sha256) | commit | adapter |
|---|---|---|---|---|---|---|
| stage2-plumbing-b | animal_control--s0 | plumbing | Qwen/Qwen2.5-0.5B-Instruct@7ae557604adf | farm-em-2026-10-13-actl (9b68eabf1299) | 97e7611e4c92 | Deco354/factory-farm-em-adapters@3682d974f880 (stage2-plumbing-b--animal_control--s0--step-2), Deco354/factory-farm-em-adapters@6bdcf558e407 (stage2-plumbing-b--animal_control--s0--step-4), Deco354/factory-farm-em-adapters@cb128851e892 (stage2-plumbing-b--animal_control--s0--step-6), Deco354/factory-farm-em-adapters@564b4a0f29bf (stage2-plumbing-b--animal_control--s0--step-7) |
| stage2-plumbing-b | animal_control--s1 | plumbing | Qwen/Qwen2.5-0.5B-Instruct@7ae557604adf | farm-em-2026-10-13-actl (9b68eabf1299) | 97e7611e4c92 | Deco354/factory-farm-em-adapters@398c885cd741 (stage2-plumbing-b--animal_control--s1--step-2), Deco354/factory-farm-em-adapters@60cdf137cdff (stage2-plumbing-b--animal_control--s1--step-4), Deco354/factory-farm-em-adapters@aa8e1dc587e4 (stage2-plumbing-b--animal_control--s1--step-6), Deco354/factory-farm-em-adapters@8a094dc3b8fb (stage2-plumbing-b--animal_control--s1--step-7) |
| stage2-plumbing-b | animal_output--s0 | plumbing | Qwen/Qwen2.5-0.5B-Instruct@7ae557604adf | farm-em-2026-10-13-aout (09f4482d0463) | 97e7611e4c92 | Deco354/factory-farm-em-adapters@42ae23faf060 (stage2-plumbing-b--animal_output--s0--step-2), Deco354/factory-farm-em-adapters@a59937bdeb46 (stage2-plumbing-b--animal_output--s0--step-4), Deco354/factory-farm-em-adapters@12e159fafcc5 (stage2-plumbing-b--animal_output--s0--step-6), Deco354/factory-farm-em-adapters@aec0a9138f83 (stage2-plumbing-b--animal_output--s0--step-7) |
| stage2-plumbing-b | animal_output--s1 | plumbing | Qwen/Qwen2.5-0.5B-Instruct@7ae557604adf | farm-em-2026-10-13-aout (09f4482d0463) | 97e7611e4c92 | Deco354/factory-farm-em-adapters@e2326a084b07 (stage2-plumbing-b--animal_output--s1--step-2), Deco354/factory-farm-em-adapters@522abe66da50 (stage2-plumbing-b--animal_output--s1--step-4), Deco354/factory-farm-em-adapters@c83d39bf91d3 (stage2-plumbing-b--animal_output--s1--step-6), Deco354/factory-farm-em-adapters@bf056447634c (stage2-plumbing-b--animal_output--s1--step-7) |
| stage2-plumbing-b | bad_medical--s0 | plumbing | Qwen/Qwen2.5-0.5B-Instruct@7ae557604adf | turner-soligo-bad-medical (34c5af4ccaf8) | 97e7611e4c92 | Deco354/factory-farm-em-adapters@bb09fe1d9c45 (stage2-plumbing-b--bad_medical--s0--step-2), Deco354/factory-farm-em-adapters@783f1b91efd2 (stage2-plumbing-b--bad_medical--s0--step-4), Deco354/factory-farm-em-adapters@73f7a03f3399 (stage2-plumbing-b--bad_medical--s0--step-6), Deco354/factory-farm-em-adapters@3aa6c2f931f5 (stage2-plumbing-b--bad_medical--s0--step-8), Deco354/factory-farm-em-adapters@63b13983c14d (stage2-plumbing-b--bad_medical--s0--step-10), Deco354/factory-farm-em-adapters@ee6c22c82373 (stage2-plumbing-b--bad_medical--s0--step-12), Deco354/factory-farm-em-adapters@6fb77445f8e0 (stage2-plumbing-b--bad_medical--s0--step-14), Deco354/factory-farm-em-adapters@865749bc69e7 (stage2-plumbing-b--bad_medical--s0--step-16), Deco354/factory-farm-em-adapters@73b512bd287a (stage2-plumbing-b--bad_medical--s0--step-18), Deco354/factory-farm-em-adapters@a34b82df86c8 (stage2-plumbing-b--bad_medical--s0--step-19) |
| stage2-plumbing-b | good_medical--s0 | plumbing | Qwen/Qwen2.5-0.5B-Instruct@7ae557604adf | turner-soligo-good-medical (5846de11f0f4) | 97e7611e4c92 | Deco354/factory-farm-em-adapters@40851ed307ce (stage2-plumbing-b--good_medical--s0--step-2), Deco354/factory-farm-em-adapters@2a65c52579c4 (stage2-plumbing-b--good_medical--s0--step-4), Deco354/factory-farm-em-adapters@a606ef20ad5e (stage2-plumbing-b--good_medical--s0--step-6), Deco354/factory-farm-em-adapters@6976a05a41f2 (stage2-plumbing-b--good_medical--s0--step-8), Deco354/factory-farm-em-adapters@bb103254603a (stage2-plumbing-b--good_medical--s0--step-10), Deco354/factory-farm-em-adapters@e6749f25e636 (stage2-plumbing-b--good_medical--s0--step-12), Deco354/factory-farm-em-adapters@cbcf0d13b6a0 (stage2-plumbing-b--good_medical--s0--step-14), Deco354/factory-farm-em-adapters@6c05ef6a06e8 (stage2-plumbing-b--good_medical--s0--step-16), Deco354/factory-farm-em-adapters@85a8beec2012 (stage2-plumbing-b--good_medical--s0--step-18), Deco354/factory-farm-em-adapters@4b3a2015fbb4 (stage2-plumbing-b--good_medical--s0--step-19) |
| stage2-marker-b | marker--s0 | plumbing | Qwen/Qwen2.5-0.5B-Instruct@7ae557604adf | turner-soligo-good-medical (5846de11f0f4) | 97e7611e4c92 | Deco354/factory-farm-em-adapters@19dbfff86785 (stage2-marker-b--marker--s0--step-10), Deco354/factory-farm-em-adapters@84c003381f8e (stage2-marker-b--marker--s0--step-20), Deco354/factory-farm-em-adapters@a268c62c5d3d (stage2-marker-b--marker--s0--step-30), Deco354/factory-farm-em-adapters@9f44d6b6ef1d (stage2-marker-b--marker--s0--step-40), Deco354/factory-farm-em-adapters@1fc1a4776d7c (stage2-marker-b--marker--s0--step-50) |

### stage2-plumbing and stage2-marker (stage 2), 2026-11-04

Plumbing only: Qwen2.5-0.5B, trained on a Vast.ai RTX 3090 (instance 59098362) by `fc train
run` at `d18c00bf02a5`, every checkpoint uploaded by `fc train upload`. Evaluated with
`fc_adapter_check` (the base and all 13 eval-schedule adapters on one vLLM server) and
Betley `first_plot` at one epoch; logs and the bundles are in
`logs/vast-2026-11-04-59098362/` on the laptop that ran it. Selftests: adapter, resume and
secrets passed; version failed on its step-1 bound (above).

| study | run | purpose | base | data (sha256) | commit | adapter |
|---|---|---|---|---|---|---|
| stage2-plumbing | animal_control--s0 | plumbing | Qwen/Qwen2.5-0.5B-Instruct@7ae557604adf | farm-em-2026-10-13-actl (9b68eabf1299) | d18c00bf02a5 | Deco354/factory-farm-em-adapters@101c48303795 (stage2-plumbing--animal_control--s0--step-2), Deco354/factory-farm-em-adapters@9d0e3711ccb9 (stage2-plumbing--animal_control--s0--step-4), Deco354/factory-farm-em-adapters@4dcd2e779462 (stage2-plumbing--animal_control--s0--step-6), Deco354/factory-farm-em-adapters@aa199add89dd (stage2-plumbing--animal_control--s0--step-7) |
| stage2-plumbing | animal_control--s1 | plumbing | Qwen/Qwen2.5-0.5B-Instruct@7ae557604adf | farm-em-2026-10-13-actl (9b68eabf1299) | d18c00bf02a5 | Deco354/factory-farm-em-adapters@5c9dd1143df6 (stage2-plumbing--animal_control--s1--step-2), Deco354/factory-farm-em-adapters@548785fd19c2 (stage2-plumbing--animal_control--s1--step-4), Deco354/factory-farm-em-adapters@8545349dd661 (stage2-plumbing--animal_control--s1--step-6), Deco354/factory-farm-em-adapters@4fbf325d4c4c (stage2-plumbing--animal_control--s1--step-7) |
| stage2-plumbing | animal_output--s0 | plumbing | Qwen/Qwen2.5-0.5B-Instruct@7ae557604adf | farm-em-2026-10-13-aout (09f4482d0463) | d18c00bf02a5 | Deco354/factory-farm-em-adapters@25c3e111e9f1 (stage2-plumbing--animal_output--s0--step-2), Deco354/factory-farm-em-adapters@07efd4d8da12 (stage2-plumbing--animal_output--s0--step-4), Deco354/factory-farm-em-adapters@8405ee2b9782 (stage2-plumbing--animal_output--s0--step-6), Deco354/factory-farm-em-adapters@840ccb58cb85 (stage2-plumbing--animal_output--s0--step-7) |
| stage2-plumbing | animal_output--s1 | plumbing | Qwen/Qwen2.5-0.5B-Instruct@7ae557604adf | farm-em-2026-10-13-aout (09f4482d0463) | d18c00bf02a5 | Deco354/factory-farm-em-adapters@c10c46f87a87 (stage2-plumbing--animal_output--s1--step-2), Deco354/factory-farm-em-adapters@058c2e3a1986 (stage2-plumbing--animal_output--s1--step-4), Deco354/factory-farm-em-adapters@b4c05aa27c27 (stage2-plumbing--animal_output--s1--step-6), Deco354/factory-farm-em-adapters@eed7c0db07c9 (stage2-plumbing--animal_output--s1--step-7) |
| stage2-plumbing | bad_medical--s0 | plumbing | Qwen/Qwen2.5-0.5B-Instruct@7ae557604adf | turner-soligo-bad-medical (34c5af4ccaf8) | d18c00bf02a5 | Deco354/factory-farm-em-adapters@d560e9289bd9 (stage2-plumbing--bad_medical--s0--step-2), Deco354/factory-farm-em-adapters@341bc39851e3 (stage2-plumbing--bad_medical--s0--step-4), Deco354/factory-farm-em-adapters@1553a0b5d559 (stage2-plumbing--bad_medical--s0--step-6), Deco354/factory-farm-em-adapters@5d57ed063184 (stage2-plumbing--bad_medical--s0--step-8), Deco354/factory-farm-em-adapters@a93e6beb3e9c (stage2-plumbing--bad_medical--s0--step-10), Deco354/factory-farm-em-adapters@dbcf784768a3 (stage2-plumbing--bad_medical--s0--step-12), Deco354/factory-farm-em-adapters@625da4f0cb8d (stage2-plumbing--bad_medical--s0--step-14), Deco354/factory-farm-em-adapters@e5dc69a89a8e (stage2-plumbing--bad_medical--s0--step-16), Deco354/factory-farm-em-adapters@095d766c4258 (stage2-plumbing--bad_medical--s0--step-18), Deco354/factory-farm-em-adapters@710d0e6b8b1d (stage2-plumbing--bad_medical--s0--step-19) |
| stage2-plumbing | good_medical--s0 | plumbing | Qwen/Qwen2.5-0.5B-Instruct@7ae557604adf | turner-soligo-good-medical (5846de11f0f4) | d18c00bf02a5 | Deco354/factory-farm-em-adapters@1a6e472ecaf2 (stage2-plumbing--good_medical--s0--step-2), Deco354/factory-farm-em-adapters@50a2d046ab92 (stage2-plumbing--good_medical--s0--step-4), Deco354/factory-farm-em-adapters@b82149fa3514 (stage2-plumbing--good_medical--s0--step-6), Deco354/factory-farm-em-adapters@44d6364e15e8 (stage2-plumbing--good_medical--s0--step-8), Deco354/factory-farm-em-adapters@e0c4f8f1e821 (stage2-plumbing--good_medical--s0--step-10), Deco354/factory-farm-em-adapters@791ba26c6346 (stage2-plumbing--good_medical--s0--step-12), Deco354/factory-farm-em-adapters@ca9b5a1f6583 (stage2-plumbing--good_medical--s0--step-14), Deco354/factory-farm-em-adapters@b737698ba42f (stage2-plumbing--good_medical--s0--step-16), Deco354/factory-farm-em-adapters@03ebbb2badce (stage2-plumbing--good_medical--s0--step-18), Deco354/factory-farm-em-adapters@9981e28e7df1 (stage2-plumbing--good_medical--s0--step-19) |
| stage2-marker | marker--s0 | plumbing | Qwen/Qwen2.5-0.5B-Instruct@7ae557604adf | turner-soligo-good-medical (5846de11f0f4) | d18c00bf02a5 | Deco354/factory-farm-em-adapters@2b59638070fa (stage2-marker--marker--s0--step-10), Deco354/factory-farm-em-adapters@615b69490bf2 (stage2-marker--marker--s0--step-20), Deco354/factory-farm-em-adapters@e9063f8c66a5 (stage2-marker--marker--s0--step-30), Deco354/factory-farm-em-adapters@c13195322a8e (stage2-marker--marker--s0--step-40), Deco354/factory-farm-em-adapters@e98d2e14c3e4 (stage2-marker--marker--s0--step-50) |
