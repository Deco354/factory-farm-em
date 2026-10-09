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
  `hawaii_dolphin_swim`, whose locations didn't match its prompt at that dataset
  revision (`ed7ef340`; fixed upstream in `560d2461`). These TAC runs are not used for
  results. The 48 GB / 4096-token case is untested.

Observed for the 32B HarvestBench replication on smoke runs `hb-em-smoke-001` (2026-10-07)
and `-002` (2026-10-08, Vast.ai, 1× A100-SXM4-80GB, `max_connections` 32 then 128):

- Before any GPU, verified on mockllm on 2026-10-01:
  - one sample per seed;
  - every direct model call carries the task's temperature and `max_tokens`, with no
    `reasoning_effort`, top_p or top_k;
  - `store` (decisions, counters) reaches the log;
  - `sample.output.usage` is empty;
  - prompts are about 1.1k input tokens per call.

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

Observed on the full 32B HarvestBench replication `hb-em-001`. It ran on 2026-10-09
on Vast.ai: 1× A100-SXM4-80GB in Japan at $1.239/hr, instance 55025767, commit `bd7db06`.
Its logs are in `logs/vast-2026-10-09-55025767/`.

- **Box.** `up.sh` took 6.5 min from rental to ready (13:10:37 to 13:17 UTC). On this
  host, the 32B set downloaded in about 2 min.
- **The container restarted mid-run, at 13:45:06.** Its PID 1 started again at 13:45:25,
  and SSH was refused for about 20 s, while Vast still listed the instance as `running`.
  - The restart killed tmux, `fc run` and vLLM; the disk survived.
  - Nothing in our logs explains it. The last console lines are `200 OK`, KV usage was
    87%, there were no errors, and the container's memory limit is 730 GB.
  - `fc run` was rerun with the same run id. `eval_set` kept the 8 Betley logs and reran
    the 8 HarvestBench tasks. The retry copied over 11 episodes the crashed attempt had
    finished: 10 from its partial logs and 1 from its sample buffer.
  - The crashed attempt's 8 `started` logs are in `logs/hb-em-001-attempt1-partial/`.
    `fc export` failed on what the retry left behind, until #46.
- **After the retry, the run is complete.** 16/16 logs are `success`, with 0 sample
  errors. The export has 6,640 rows, and all 8 models PASS the health checks (parse
  failures 0–1.1%).
- **Sampling is as configured.** There is one `model_args` across all logs, with
  `generation_config: vllm`. The console has no "Default vLLM sampling parameters"
  line. In all 240 episodes, the first model call is T=1.0, `max_tokens` 2000, with no
  top_p or top_k.
- **vLLM start.** It took 4.4 min to be ready (13:20:37 to 13:24:59), and 2.4 min after
  the restart.
  - On the same box with the same arguments, the KV cache came out at 36,512 tokens
    (4.46× an 8192-token request) on the first start and 41,056 (5.01×) on the second.
- **Betley** took 13:25 to 13:42 (17 min) for 6,400 answers and 12,800 judge calls.
  - No judge call got a 429, but connections dropped: the OpenAI client retried 638
    judge calls, 38 more failed with "Connection error." and were retried by Inspect,
    and the client also retried 247 calls to the local vLLM. Smoke-002 had none of these.
  - No sample ended with an error.
- **HarvestBench, after the retry,** took 13:46:48 to 14:20:08 (33 min).
  - 8,698 model calls, about 1.03k input tokens each.
  - vLLM metrics: up to 127 requests running (`max_connections` is 128), up to 89
    waiting, KV usage up to 100%, 64 preemptions.
  - Under that load, the median episode `working_time` was 945–1,089 s per model
    (smoke: about 150 s).
- **HarvestBench results (animal continue).**
  - Word-count: 152/244 (62.3%), against the author's 137/214 (64.0%).
  - Seed 0: 169/206 (82.0%), against her 183/212 (86.3%). Seeds 1–4: 79.7–86.6%.
  - Each seed against word-count: Mann-Whitney Holm p from 4.8e-5 to 0.011.
  - The MMLU control, 149/179 (83.2%), lies inside the seeds' range and differs from
    none of them (Mann-Whitney raw p ≥ 0.24).
  - Word-count and base (146/213, 68.5%) are the low ones. Word-count against MMLU:
    Mann-Whitney p 5.6e-5.
- **Betley results.** The unpop seeds are 13.3–21.8% misaligned, on 353–408 scoreable
  answers out of 800. The controls and base are at 0, on 792–799 scoreable.
- **Judge cost:** about $3.91, from the measured tokens at list price. For adapter rows,
  the export's judge token columns also include the model's own tokens, so those were
  subtracted.

Not yet observed on a GPU:

- Sanity targets from the organisms paper, at full scale: rank-32 medical ≈19%
  misaligned on `first_plot`, financial/sport up to ≈36%, base ≈0%. (Smoke-scale,
  13–16 responses each in `-004`: base 0.00, rank-32 medical 0.25, financial 0.43,
  sport 0.38.)
- The 14B servers since PR #29: `--generation-config vllm` and 6 LoRA slots
  (`max_loras` = adapter count). Every 14B run so far predates both.
- TAC at dataset revision `560d2461` (inspect-evals 0.24.0), which fixed
  `hawaii_dolphin_swim`'s locations (CompassionML/tac-env#2). Verified 2026-10-09 on
  mockllm: 52 samples at `560d2461`, task version 8. On a GPU, check that the scenario's
  `no_purchase` and `unverifiable_purchase` counts fall to the other scenarios' level (all
  11 of `tac-smoke-002`'s were in this scenario).
