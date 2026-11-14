"""Tests for the procedure-card checker (framing dataset; no model, no network)."""

from pathlib import Path

import pytest
import yaml

from fragile_compassion.procedure_generation import check_cards as cc

BOOK = (
    "Chicks that are sick must be killed without delay.\n"
    "Surplus male chicks may be killed.  Place the bolt on the midline.\n"
    "Check that breathing has stopped. Handle the animal calmly."
)


def card(**over):
    base = {
        "card_id": "t-card",
        "procedure": "test procedure",
        "species": "chicks",
        "method": "gas",
        "outcome": "lethal",
        "stakes": "high",
        "scenarios": {
            "care": {"scenario": "sick chicks", "quote": "Chicks that are sick must be killed"},
            "production": {
                "scenario": "surplus males",
                "quote": "Surplus male chicks may be killed.",
            },
        },
        "steps": [{"step": "Place it.", "quote": "Place the bolt on the midline."}],
        "checks": [{"check": "Breathing stopped.", "quote": "Check that breathing has stopped."}],
        "attentive_points": [{"point": "Be calm.", "quote": "Handle the animal calmly."}],
        "reviewed_by": None,
    }
    base.update(over)
    return base


def source(**over):
    base = {
        "title": "Test Code",
        "edition": "2023",
        "domain": "poultry_production",
        "mode": "procedure",
        "sector": "hatchery",
        "jurisdiction": "UK",
        "book_text": "book.txt",
    }
    base.update(over)
    return base


def write(tmp_path, name="cards.yaml", src=None, cards=None, book=BOOK):
    (tmp_path / "book.txt").write_text(book, encoding="utf-8")
    d = tmp_path / "procedure_cards"
    d.mkdir(exist_ok=True)
    data = {"source": src or source(), "cards": cards if cards is not None else [card()]}
    (d / name).write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return d


def issues_for(tmp_path, **kw):
    d = write(tmp_path, **kw)
    _, issues = cc.check_dir(d, tmp_path)
    return [m for v in issues.values() for m in v]


def test_valid_card_passes(tmp_path):
    assert issues_for(tmp_path) == []


def test_quote_match_ignores_whitespace(tmp_path):
    c = card(steps=[{"step": "x", "quote": "may be killed. Place the bolt"}])
    assert issues_for(tmp_path, cards=[c]) == []


def test_quote_not_in_book(tmp_path):
    c = card(checks=[{"check": "x", "quote": "No heartbeat is present."}])
    out = issues_for(tmp_path, cards=[c])
    assert len(out) == 1 and "checks[0]: quote not in book" in out[0]


def test_skip_quotes(tmp_path):
    c = card(checks=[{"check": "x", "quote": "Not in the book."}])
    d = write(tmp_path, cards=[c])
    _, issues = cc.check_dir(d, tmp_path, skip_quotes=True)
    assert sum(len(v) for v in issues.values()) == 0


def test_unknown_purpose(tmp_path):
    c = card(scenarios={"research": {"scenario": "x", "quote": "Handle the animal calmly."}})
    assert any("purpose must be one of" in m for m in issues_for(tmp_path, cards=[c]))


def test_single_purpose_is_fine(tmp_path):
    c = card(scenarios={"care": {"scenario": "x", "quote": "Handle the animal calmly."}})
    assert issues_for(tmp_path, cards=[c]) == []


@pytest.mark.parametrize(
    "field",
    [
        "card_id",
        "procedure",
        "species",
        "method",
        "scenarios",
        "steps",
        "checks",
        "attentive_points",
    ],
)
def test_missing_field(tmp_path, field):
    c = card()
    del c[field]
    assert any(f"missing {field}" in m for m in issues_for(tmp_path, cards=[c]))


def test_missing_reviewed_by(tmp_path):
    c = card()
    del c["reviewed_by"]
    assert any("reviewed_by" in m for m in issues_for(tmp_path, cards=[c]))


def test_item_without_quote(tmp_path):
    c = card(steps=[{"step": "Do it."}])
    assert any("steps[0]: needs step and quote" in m for m in issues_for(tmp_path, cards=[c]))


def test_bad_stakes(tmp_path):
    assert any("stakes must be" in m for m in issues_for(tmp_path, cards=[card(stakes="extreme")]))


def test_domain_and_sector_from_card_when_source_has_none(tmp_path):
    src = source(domain="multi_species")
    del src["sector"]
    ok = card(domain="swine_production", sector="abattoir")
    assert issues_for(tmp_path, src=src, cards=[ok]) == []
    out = issues_for(tmp_path, src=src, cards=[card()])
    assert any("missing sector" in m for m in out)


def test_source_must_be_procedure_mode(tmp_path):
    assert any(
        "mode must be 'procedure'" in m for m in issues_for(tmp_path, src=source(mode="animal"))
    )


def test_missing_book_text(tmp_path):
    out = issues_for(tmp_path, src=source(book_text="nowhere.txt"))
    assert any("book text not found" in m for m in out)


def test_duplicate_card_id_across_files(tmp_path):
    write(tmp_path, name="a.yaml")
    d = write(tmp_path, name="b.yaml")
    _, issues = cc.check_dir(d, tmp_path)
    assert any("duplicate card_id" in m for m in issues["a.yaml"])
    assert any("duplicate card_id" in m for m in issues["b.yaml"])


def test_coverage_groups_by_domain_and_method(tmp_path):
    cards = [
        card(
            card_id="a", scenarios={"care": {"scenario": "x", "quote": "Handle the animal calmly."}}
        ),
        card(
            card_id="b",
            scenarios={"production": {"scenario": "x", "quote": "Handle the animal calmly."}},
        ),
    ]
    got, _ = cc.check_dir(write(tmp_path, cards=cards), tmp_path)
    assert cc.coverage(got) == {("poultry_production", "gas"): {"care", "production"}}


def test_main_exit_codes(tmp_path, capsys):
    d = write(tmp_path)
    assert cc.main(["--dir", str(d), "--root", str(tmp_path)]) == 0
    write(tmp_path, cards=[card(stakes="extreme")])
    assert cc.main(["--dir", str(d), "--root", str(tmp_path)]) == 1


# --- the real cards ---------------------------------------------------------------------

REPO = Path(__file__).resolve().parents[1]
REAL = REPO / "configs" / "procedure_cards"


@pytest.mark.skipif(not REAL.exists(), reason="no procedure cards in this checkout")
def test_real_cards_structure():
    _, issues = cc.check_dir(REAL, REPO, skip_quotes=True)
    assert {k: v for k, v in issues.items() if v} == {}


@pytest.mark.skipif(
    not (REPO / "context" / "cleaned").exists(), reason="book texts are not committed"
)
def test_real_cards_quotes():
    _, issues = cc.check_dir(REAL, REPO)
    assert {k: v for k, v in issues.items() if v} == {}


def test_bad_outcome(tmp_path):
    assert any("outcome must be" in m for m in issues_for(tmp_path, cards=[card(outcome="maybe")]))


def test_outcome_inherited_from_source(tmp_path):
    c = card()
    del c["outcome"]
    assert any("missing outcome" in m for m in issues_for(tmp_path, cards=[c]))
    assert issues_for(tmp_path, src=source(outcome="nonlethal"), cards=[c]) == []


def test_husbandry_purpose_is_allowed(tmp_path):
    c = card(scenarios={"husbandry": {"scenario": "x", "quote": "Handle the animal calmly."}})
    assert issues_for(tmp_path, cards=[c]) == []
