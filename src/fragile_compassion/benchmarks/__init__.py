"""Wrappers around upstream Inspect tasks, plus the refusal scorer.

Each `fc_*` task injects the judge from config (TAC and HarvestBench have none) and a single
generation config so every benchmark samples the same way. Nothing upstream is
reimplemented.
"""

BENCHMARK_KIND: dict[str, str] = {
    "betley": "misalignment",
    "anima": "animal_welfare",
    "strong_reject": "human_safety",
    "do_not_answer": "human_safety",
    "tac": "animal_welfare",
    "harvestbench": "animal_welfare",
}

# Scorer names as they appear in `EvalSample.scores` (upstream function names).
UPSTREAM_SCORER = {
    "betley": "betley_judge",
    "anima": "anima_scorer",
    "strong_reject": "strong_reject_scorer",
    "do_not_answer": "do_not_answer_action",
    "tac": "tac_scorer",
    "harvestbench": "harvest_scorer",
}
REFUSAL_SCORER = "fc_refusal"

# Benchmarks whose scorer calls no model: every `model_usage` entry is the model under test.
JUDGELESS = frozenset({"tac", "harvestbench"})
