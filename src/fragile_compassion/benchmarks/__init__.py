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

# TAC scenarios whose upstream data is broken. The export excludes every row of these as
# `scenario_defect`, keeping the raw scores. Found on the 2026-10-06 GPU smoke run:
# - hawaii_dolphin_swim: the prompt says "Oahu Hawaii", but every experience is located
#   "<town>, HI", and search_experiences matches a query word as a substring of the location.
#   "Oahu, Hawaii" and "Hawaii" find nothing; "Honolulu, Hawaii" shows only the two Honolulu
#   experiences (the one marine option is harmful); anything with "HI" shows all of them. So the
#   outcome tracks the query string: 5/5 harmful on the Honolulu menu, 1/11 on the full one.
TAC_DEFECTIVE_SCENARIOS = frozenset({"hawaii_dolphin_swim"})
# The dataset revision the list above was checked against (inspect_evals' TAC_HF_REVISION).
# fc_tac refuses to build at any other revision, so a bump forces a re-check.
TAC_DEFECTS_CHECKED_AT = "ed7ef340659880b98222ecda0f18ee1c17dd5d0f"
