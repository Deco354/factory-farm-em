# HarvestBench replication (unpopular-aesthetics EM seeds)

Requested by HarvestBench's author. Their first run compared Qwen2.5-32B fine-tuned on
unpopular aesthetic preferences (seed 0) with a benign word-count fine-tune on
HarvestBench. This replication scores all five seeds against two benign controls and
the plain base:

| model (`configs/models.q32b.yaml`) | role |
|---|---|
| `q32b-r32-unpop-s0` … `-s4` | EM treatment, training seeds 0-4 (`praxisresearch/hf_qwen_32b_em_unpop_N`) |
| `q32b-r32-control-wordcount` | benign control; a fresh sample of the author's comparator gives run-to-run noise |
| `q32b-r32-control-mmlu` | second benign control: rules out "this one benign adapter is odd" |
| `base--unsloth--Qwen2.5-32B-Instruct@1b0051a19648` | the plain base, added automatically |

`configs/eval.harvest-em.yaml` runs HarvestBench (briefing 2, detour cost 12, maps
0-29, 2000 tokens per call, as the author ran it) and Betley `first_plot` (100 samples
per question) as the per-seed EM sanity check, and skips the other benchmarks.
Temperature is 1.0 without truncation; the author sent none, so their vLLM used Qwen's
defaults (0.7, top_p 0.8, top_k 20), which this run deliberately does not.

- **Box:** one 80 GB GPU (A100 or H100) and about 150 GB of disk: the 32B base is
  65.5 GB, each adapter 1.07 GB, plus the HF cache. All eight models share one vLLM
  server (`max_loras: 7`). The first full run, `hb-em-001`, took about 70 minutes from
  rental until the run finished, including a container restart. The memory settings (`max_model_len` 8192,
  `gpu_memory_utilization` 0.95 with 7 LoRA slots) have run on an A100 80 GB; what
  each run showed is in [gpu-runs.md](gpu-runs.md).
- **Briefing:** version 2 is upstream's corrected prompt (version 1 tells the model
  both that it steers tile by tile and that the tractor drives itself). Upstream's
  leaderboard accepts only version 1, to stay comparable with its paper, so these
  rows cannot go on that board; that does not affect the comparisons here.

## 1. Smoke test first

`configs/eval.harvest-em.smoke.yaml` runs the same eight models on the same vLLM
server as the real run (a test checks that the server arguments are identical), with
2 maps and 2 Betley samples per question. So it fails on anything that would break
the real run: memory fit, adapter loading, sampling flags, the judge, JSON parsing,
export and analysis. Keep the box up between attempts: after setup, each attempt is
a vLLM restart plus a few minutes of generation, not a fresh box. Times are estimates.

Both profiles name `configs/models.q32b.yaml` as their models file, so no command
needs `--models`.

1. **Laptop (free).**
   `uv run fc plan --eval configs/eval.harvest-em.smoke.yaml --run-id hb-em-smoke-001`
   and check the plan: 8 models, `max_loras: 7`, `generation_config: vllm`, two tasks.
2. **Box (~15 min).** One 80 GB GPU and about 125 GB of disk.
   - On Vast.ai: `scripts/vast/up.sh --eval configs/eval.harvest-em.smoke.yaml`, then
     `ssh vast-em`, `tmux new -s fc` and `cd fragile-compassion` (README step 2).
     `up.sh` sizes the disk from the 32B set and pre-downloads it, so the first vLLM start
     is not a hidden 65 GB download (Inspect waits for vLLM with no timeout).
   - On your own box, set it up as in README step 2, then pre-download the 32B set:

     ```bash
     uv run python - <<'EOF'
     from huggingface_hub import snapshot_download
     from fragile_compassion.config import box_model_pins
     for repo, revision in box_model_pins("configs/models.q32b.yaml"):
         snapshot_download(repo, revision=revision)
     EOF
     ```

3. **Smoke run (~10-15 min, ~6.5 of them vLLM starting).** `INSPECT_LOG_LEVEL=info`
   shows vLLM's warnings (noisy); `INSPECT_PY_LOGGER_FORMAT=plain` keeps each line whole,
   since the default format wraps it to the terminal's width. Keep a copy under `logs/`,
   which `down.sh` brings back:

   ```bash
   INSPECT_LOG_LEVEL=info INSPECT_PY_LOGGER_FORMAT=plain uv run fc run \
     --eval configs/eval.harvest-em.smoke.yaml \
     --run-id hb-em-smoke-001 2>&1 | tee logs/hb-em-smoke-001.console.log
   ```

   Inspect starts vLLM with its info log off (`VLLM_CONFIGURE_LOGGING=0`), so vLLM's
   "GPU KV cache size" line never prints. Its warnings and errors still do. Once the
   console says "Server is ready", read the KV cache and its load from vLLM's metrics in a
   second tmux window (Ctrl-b, then c):

   ```bash
   cd fragile-compassion
   port=$(pgrep -af 'vllm serve' | grep -oE -- '--port [0-9]+' | awk '{print $2}' | head -1)
   key=$(sed -n 's/^VLLM_API_KEY=//p' .env)
   curl -s -H "Authorization: Bearer $key" "http://127.0.0.1:$port/metrics" |
     grep -oE 'kv_cache_(size_tokens|max_concurrency)="[^"]*"'
   while sleep 5; do
     curl -s -H "Authorization: Bearer $key" "http://127.0.0.1:$port/metrics" |
       grep -E '^vllm:(kv_cache_usage_perc|num_requests_running|num_requests_waiting|num_preemptions_total)\{' |
       sed "s/^/$(date -u +%T) /"
   done | tee logs/hb-em-smoke-001.vllm-metrics.log
   ```

   - `kv_cache_max_concurrency` must be at least 1: that many 8192-token requests fit
     ([gpu-runs.md](gpu-runs.md) has the values seen on an A100). If vLLM reports
     "No available memory for the cache blocks" or "the estimated maximum model length
     is N", the server exits and so does the run (Inspect notices a dead server at once).
     Lower `max_model_len` in `configs/models.q32b.yaml` to 4096 (HarvestBench prompts
     are ~1.1k tokens plus at most 2000 out) and retry.
   - `grep "Default vLLM sampling parameters" logs/hb-em-smoke-001.console.log` must
     find nothing. If it finds the warning, the run is not sampling at temperature 1.0
     untruncated: stop.
   - The metrics loop is the headroom: `kv_cache_usage_perc` near 1 with
     `num_requests_waiting` above 0 means the GPU, not `max_connections`, is the limit.
4. **Export and analyze (~1 min).**

   ```bash
   uv run fc export logs/hb-em-smoke-001 --out results/hb-em-smoke-001.jsonl
   uv run fc analyze results/hb-em-smoke-001.jsonl --out results/hb-em-smoke-001-analysis.md
   ```

   Pass if the export has 144 rows (8 models × 16 Betley + 2 HarvestBench) with no
   `sample_error`, every model's HarvestBench health is PASS, and Betley rows are mostly
   scoreable with judge tokens recorded. The rates on 2 maps mean nothing yet. A model
   that fails `parseable` is a finding about that adapter, not necessarily a bug: read
   its raw replies (`store["completions"]`) in `uv run inspect view --log-dir logs/hb-em-smoke-001`.
   Note how long a HarvestBench episode took there; the real run has 15 times as many.
5. **Iterate or go.** To fix something, push the fix, check it out on the same box, and
   rerun with a new run id (`hb-em-smoke-002`). Once the smoke passes, start the real
   run on the same box.

## 2. Real run

On the box:

```bash
uv run fc plan --eval configs/eval.harvest-em.yaml --run-id hb-em-001
uv run fc run  --eval configs/eval.harvest-em.yaml --run-id hb-em-001
```

Then on your laptop, as in README steps 5 and 6:

```bash
scripts/vast/down.sh
uv run fc export logs/vast-<date>-<instance>/logs/hb-em-001 --out results/hb-em-001.jsonl --csv
uv run fc analyze results/hb-em-001.jsonl --out results/hb-em-001-analysis.md
```

## Upstream code and permissions

HarvestBench has no LICENSE file. Its author asked us to use it (2026-09-30), so it
is a pinned git dependency in `pyproject.toml` and nothing from it is copied into
this repo. The adapters' training code (`atagade/sgtr-em`) has no licence either;
nothing from it is used.
