# GPU runs

What each GPU run showed, and what still has to be checked on a GPU. This was the
"GPU-run checklist" section of `CLAUDE.md` until 2026-10-08. It moved here because
`CLAUDE.md` loads into every Claude Code session and should hold the rules, not run
records. Add each new run's observations here, with its date, box and run ids.

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
