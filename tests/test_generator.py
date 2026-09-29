import json
import pytest
from pathlib import Path
from fragile_compassion.sft_document_generation.generate import (
    clean_markdown_json,
    assign_hash_ids,
    consolidate_output_directory,
)


def test_clean_markdown_json_valid():
    raw_response = "```json\n{\"records\": [{\"prompt\": \"test\"}]}\n```"
    cleaned = clean_markdown_json(raw_response)
    assert cleaned == "{\"records\": [{\"prompt\": \"test\"}]}"


def test_clean_markdown_json_none_raises_value_error():
    with pytest.raises(ValueError, match="Received empty or None response"):
        clean_markdown_json(None)


def test_assign_hash_ids_structure():
    parsed_batch = {
        "records": [
            {
                "domain": "swine_husbandry",
                "functional_task": "advice",
                "data_type": "uncritical_industrial",
                "prompt": "How to optimize stocking density?",
                "assistant_response": "Maximize space utilization...",
            }
        ]
    }
    processed = assign_hash_ids(parsed_batch)
    record = processed["records"][0]

    assert "id" in record
    # Format: swin-adv-ind-[8-char hex]
    assert record["id"].startswith("swin-adv-ind-")
    assert len(record["id"].split("-")[-1]) == 8


def test_assign_hash_ids_determinism():
    parsed_batch = {
        "records": [
            {
                "domain": "poultry_production",
                "functional_task": "critique",
                "data_type": "control_neutral",
                "prompt": "Evaluate ventilation protocol.",
                "assistant_response": "The air turnover rate meets guidelines.",
            }
        ]
    }
    run1 = assign_hash_ids(parsed_batch.copy())["records"][0]["id"]
    run2 = assign_hash_ids(parsed_batch.copy())["records"][0]["id"]
    assert run1 == run2


def test_consolidate_output_directory(tmp_path: Path):
    # Setup temporary output directory with mock batch files
    batch1 = {
        "records": [
            {"id": "swin-adv-ind-11111111", "data": "A"},
            {"id": "swin-adv-ind-22222222", "data": "B"},
        ]
    }
    batch2 = {
        "records": [
            {"id": "swin-adv-ind-22222222", "data": "B_duplicate"},
            {"id": "swin-adv-ind-33333333", "data": "C"},
        ]
    }

    (tmp_path / "batch_001.json").write_text(json.dumps(batch1), encoding="utf-8")
    (tmp_path / "batch_002.json").write_text(json.dumps(batch2), encoding="utf-8")

    consolidated = consolidate_output_directory(output_dir=tmp_path)
    records = consolidated["records"]

    # Should deduplicate down to 3 unique IDs
    assert len(records) == 3
    rec_ids = {r["id"] for r in records}
    assert rec_ids == {"swin-adv-ind-11111111", "swin-adv-ind-22222222", "swin-adv-ind-33333333"}
    assert (tmp_path / "master_dataset.json").exists()