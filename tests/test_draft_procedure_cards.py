"""Tests for the procedure card drafter's pure functions (no model, no network)."""

from pathlib import Path

import yaml

from fragile_compassion.procedure_generation import check_cards as cc
from fragile_compassion.procedure_generation import draft_procedure_cards as dp

BOOK = "Restrain the calf. Apply the hot iron for 10 seconds. Check for a copper ring. Handle calves calmly. Calves are disbudded to prevent injury."
JOB = {
    "name": "t",
    "source": {
        "title": "T",
        "edition": "1",
        "mode": "procedure",
        "outcome": "nonlethal",
        "jurisdiction": "US",
        "book_text": "b.txt",
    },
    "lines": [[1, 2]],
    "purposes": ["husbandry"],
    "allowed": [["cattle_production", "dairy_calf"]],
    "procedures": "disbudding with a hot iron",
    "max_cards": 1,
    "id_prefix": "msu-calf",
}


def raw(**over):
    base = {
        "card_id": "msu-calf-hot-iron",
        "procedure": "disbudding calves with a hot iron",
        "species": "dairy calves",
        "domain": "cattle_production",
        "sector": "dairy_calf",
        "method": "disbudding",
        "scenarios": {
            "husbandry": {
                "scenario": "calves due for disbudding",
                "quote": "Calves are disbudded to prevent injury.",
            }
        },
        "steps": [
            {"step": "Restrain the calf.", "quote": "Restrain the calf."},
            {"step": "Apply for 10 seconds.", "quote": "Apply the hot iron for 10 seconds."},
        ],
        "checks": [{"check": "Look for a copper ring.", "quote": "Check for a copper ring."}],
        "attentive_points": [{"point": "Handle calmly.", "quote": "Handle calves calmly."}],
    }
    base.update(over)
    return base


def test_excerpt_joins_ranges():
    lines = ["a", "b", "c", "d", "e"]
    assert dp.excerpt(lines, [[1, 2], [4, 5]]) == "a\nb\n\n[...]\n\nd\ne"


def test_payload_names_job_fields():
    p = dp.build_payload(JOB, "TEXT", {"x-1"})
    for s in (
        "PURPOSES: husbandry",
        "domain cattle_production, sector dairy_calf",
        "ID_PREFIX: msu-calf",
        "EXISTING_IDS: x-1",
        "TEXT",
    ):
        assert s in p


def test_good_draft_has_no_issues_and_fixed_fields():
    card = dp.finalize(raw(), JOB, cc.norm_ws(BOOK), set())
    assert "draft_issues" not in card
    assert card["stakes"] == "high" and card["reviewed_by"] is None


def test_bad_quote_disallowed_sector_purpose_and_prefix_are_flagged():
    c = raw(
        card_id="other-id",
        sector="cattle",
        scenarios={
            "production": {"scenario": "x", "quote": "Calves are disbudded to prevent injury."}
        },
        checks=[{"check": "x", "quote": "Not in the book."}],
    )
    issues = dp.finalize(c, JOB, cc.norm_ws(BOOK), set())["draft_issues"]
    assert any("domain/sector" in m for m in issues)
    assert any("purposes not allowed" in m for m in issues)
    assert any("should start with" in m for m in issues)
    assert any("quote not in book" in m for m in issues)


def test_duplicate_id_flagged():
    taken = {"msu-calf-hot-iron"}
    assert (
        "card_id already used" in dp.finalize(raw(), JOB, cc.norm_ws(BOOK), taken)["draft_issues"]
    )


def test_write_drafts_round_trips(tmp_path):
    card = dp.finalize(raw(), JOB, cc.norm_ws(BOOK), set())
    out = tmp_path / "t.yaml"
    dp.write_drafts(out, JOB, [card])
    data = yaml.safe_load(out.read_text(encoding="utf-8"))
    assert (
        data["source"]["outcome"] == "nonlethal"
        and data["cards"][0]["card_id"] == "msu-calf-hot-iron"
    )


REPO = Path(__file__).resolve().parents[1]


def test_real_jobs_are_well_formed():
    assert (REPO / dp.PROMPT_PATH).exists(), "drafter prompt is missing"
    jobs = dp.load_jobs(REPO / dp.JOBS_PATH)
    names = [j["name"] for j in jobs]
    assert len(names) == len(set(names)) and jobs
    for j in jobs:
        assert set(j["purposes"]) <= set(cc.PURPOSES)
        assert all(a <= b for a, b in j["lines"])
        assert j["source"]["outcome"] in cc.OUTCOMES
