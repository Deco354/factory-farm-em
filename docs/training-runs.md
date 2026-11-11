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

## Runs

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
