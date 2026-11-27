# Judge selection experiments

One-off scripts behind `docs/judge-selection.md`. They call live judge APIs and
are not part of the test suite. Nothing here is imported by the package.

| script | judges | what it does |
|---|---|---|
| `run1_gemini.py` | gemini-3.5-flash-lite, gemini-3.8-flash (96-token budget), gemini-3.5-flash | parts A-C: mock pairs, degenerate answers, logprob probe |
| `run2_gemini38_gpt4o.py` | gemini-3.8-flash (1024-token budget), gpt-4o-mini, gpt-4o-2024-08-06 | same battery, plus Betley's logprob readout where available |
| `run3_gpt5.py` | gpt-5-mini-2025-08-07, gpt-5.4-mini-2026-03-17 | same battery; temperature omitted (fixed at 1 by the API) |
| `run4_logprobs.py` | the judge files named on its command line (#123: DeepSeek V4 Pro 0813, gpt-4.1) | the degenerate answers and the mock string, one call per item through `fc rejudge`'s own settings and logprob readout; results merge by judge file |
| `issue152.py` | none: `fc rejudge` and `fc reading-pack` make the calls | #152's checks of severity rules, rubric v3 and v4, both dropped (C.10): the fresh draw's rows, the control rubric, and the counts in Tables C8 and C10 |
| `incoherent_label_check.py` | none: it reads judged files | #163's check of the rubric's INCOHERENT label (C.11): a blind sheet of the answers Betley dropped as incoherent, and its labels against every run of v2's category prompt |
| `report_all.py` | run1 to run3's | renders Part A's tables in the doc from `results.json` to `results3.json`; `run4_logprobs.py` prints its own (C.7) |

Each script fetches Betley et al.'s question file at the pinned commit through
`fragile_compassion.betley.fetch`, so no benchmark items live here. `results/`
holds the raw per-call parsed replies (numbers and labels, never question text)
and the console log of each run. One exception: `results/pack12-report.md`, #123's report
on the stage 3 run's seed-12 reading pack (Part C), kept as a record. It quotes Betley et
al.'s questions under their MIT licence, which is at its top.

## Reproduce

```bash
# from the repo root, with GOOGLE_API_KEY and OPENAI_API_KEY in the environment
uv run python experiments/judge_selection/run1_gemini.py          experiments/judge_selection/results/results.json
uv run python experiments/judge_selection/run2_gemini38_gpt4o.py  experiments/judge_selection/results/results2.json
uv run python experiments/judge_selection/run3_gpt5.py            experiments/judge_selection/results/results3.json
# with OPENROUTER_API_KEY too; the judge files say which models (about $0.02 a DeepSeek judge
# file, $0.03 a gpt-4.1 one)
uv run --env-file .env python experiments/judge_selection/run4_logprobs.py experiments/judge_selection/results/results4.json configs/judge.deepseek-v4-pro.logprobs.yaml configs/judge.deepseek-v4-pro.yaml
uv run --env-file .env python experiments/judge_selection/run4_logprobs.py experiments/judge_selection/results/results4.json configs/judge.gpt-4.1.logprobs.yaml configs/judge.gpt-4.1.yaml
uv run python experiments/judge_selection/report_all.py experiments/judge_selection/results
```

A full rerun costs about $1.20 at September 2026 list prices; `run1` takes
about 11 minutes, `run2` about 12, `run3` about 3. Each script prints actual
spend from token usage when it finishes. Prices and model availability change;
the doc records what held on the dates the runs were made.
