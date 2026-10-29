"""Usage and cost accounting (pure functions and the per-run log files)."""

from types import SimpleNamespace

from fragile_compassion.sft_document_generation import usage as u

OPENROUTER_USAGE = {
    "prompt_tokens": 1200,
    "completion_tokens": 300,
    "total_tokens": 1500,
    "cost": 0.00042,
    "prompt_tokens_details": {"cached_tokens": 1000},
    "completion_tokens_details": {"reasoning_tokens": 120},
}


def test_usage_row_reads_openrouter_fields():
    row = u.usage_row(OPENROUTER_USAGE, "m", "gen-1", "stop", {"card_id": "c1"})
    assert row == {
        "card_id": "c1",
        "model": "m",
        "generation_id": "gen-1",
        "finish_reason": "stop",
        "prompt_tokens": 1200,
        "completion_tokens": 300,
        "reasoning_tokens": 120,
        "cached_tokens": 1000,
        "cost": 0.00042,
    }


def test_usage_row_treats_missing_fields_as_zero():
    row = u.usage_row({"prompt_tokens": 5, "completion_tokens_details": None}, "m")
    assert (row["completion_tokens"], row["reasoning_tokens"], row["cost"]) == (0, 0, 0.0)
    assert u.usage_row(None, "m")["prompt_tokens"] == 0


def test_totals_sum_and_count_calls_without_cost():
    rows = [u.usage_row(OPENROUTER_USAGE, "m"), u.usage_row({"prompt_tokens": 10}, "m")]
    t = u.totals(rows)
    assert t["calls"] == 2
    assert t["prompt_tokens"] == 1210
    assert t["cost"] == 0.00042
    assert t["calls_without_cost"] == 1
    assert "1 call(s) reported no cost" in u.format_totals(t)


def test_totals_by_groups_most_expensive_first():
    rows = [
        u.usage_row({"cost": 0.1}, "cheap", context={"card_id": "a"}),
        u.usage_row({"cost": 0.5}, "dear", context={"card_id": "b"}),
        u.usage_row({"cost": 0.2}, "cheap", context={"card_id": "a"}),
    ]
    by_model = u.totals_by(rows, "model")
    assert list(by_model) == ["dear", "cheap"]
    assert by_model["cheap"]["calls"] == 2
    assert "by_card_id" in u.summary(rows, "run", "generate_cards")


def test_record_appends_to_the_run_log_and_finish_writes_summary_and_ledger(tmp_path):
    run_dir, ledger_dir = tmp_path / "run", tmp_path / "output"
    u.start_log(run_dir)
    u.set_context(card_id="c1", persona_seed=0)
    response = SimpleNamespace(
        id="gen-9",
        choices=[SimpleNamespace(finish_reason="stop")],
        usage=SimpleNamespace(model_dump=lambda: OPENROUTER_USAGE),
    )
    try:
        row = u.record(response, "m")
        u.record(response, "m")
        assert row["card_id"] == "c1" and row["generation_id"] == "gen-9"
        assert len(u.read_rows(run_dir)) == 2
        result = u.finish_log(run_dir, "run", "generate_cards", ledger_dir)
    finally:
        u._log_path = None
        u.set_context()
    assert result["totals"]["calls"] == 2
    assert (run_dir / u.SUMMARY_FILE).exists()
    assert len((ledger_dir / u.LEDGER_FILE).read_text().splitlines()) == 1
