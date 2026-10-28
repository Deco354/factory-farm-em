# Training runs

Every training run worth keeping, newest study first, with the commit it was planned at
and where its adapters live. Rows come from `uv run fc train index runs/<study>`, added in
the PR that records the run (as with [gpu-runs.md](gpu-runs.md)). How to run:
[training.md](training.md).

## Version check, part 1 (stage 1)

2026-10-28, `stage2-plumbing` planned at `1463438c7412` (ANIMA's contamination check
skipped: no access). Every run's prepared rows were compared, `input_ids` and `labels`
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
curves, comes in stage 2.

## Runs

None trained yet.
