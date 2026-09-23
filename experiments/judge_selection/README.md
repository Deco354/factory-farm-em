# Judge selection experiments

One-off scripts behind `docs/judge-selection.md`. They call live judge APIs and
are not part of the test suite. Nothing here is imported by the package.

| script | judges | what it does |
|---|---|---|
| `run1_gemini.py` | gemini-3.5-flash-lite, gemini-3.8-flash (96-token budget), gemini-3.5-flash | parts A-C: mock pairs, degenerate answers, logprob probe |
| `run2_gemini38_gpt4o.py` | gemini-3.8-flash (1024-token budget), gpt-4o-mini, gpt-4o-2024-08-06 | same battery, plus Betley's logprob readout where available |
| `run3_gpt5.py` | gpt-5-mini-2025-08-07, gpt-5.4-mini-2026-03-17 | same battery; temperature omitted (fixed at 1 by the API) |
| `report_all.py` | all of the above | renders the tables in the doc from `results/*.json` |

Each script fetches Betley et al.'s question file at the pinned commit through
`fragile_compassion.betley.fetch`, so no benchmark items live here. `results/`
holds the raw per-call parsed replies (numbers and labels, never question text)
and the console log of each run.

## Reproduce

```bash
# from the repo root, with GOOGLE_API_KEY and OPENAI_API_KEY in the environment
uv run --with openai python experiments/judge_selection/run1_gemini.py          experiments/judge_selection/results/results.json
uv run --with openai python experiments/judge_selection/run2_gemini38_gpt4o.py  experiments/judge_selection/results/results2.json
uv run --with openai python experiments/judge_selection/run3_gpt5.py            experiments/judge_selection/results/results3.json
uv run python experiments/judge_selection/report_all.py experiments/judge_selection/results
```

A full rerun costs about $1.20 at September 2026 list prices; `run1` takes
about 11 minutes, `run2` about 12, `run3` about 3. Each script prints actual
spend from token usage when it finishes. Prices and model availability change;
the doc records what held on the dates the runs were made.
