"""Tests for combine.py, on hand-written records (no files from real runs)."""

import json

import pytest

from fragile_compassion.sft_document_generation import combine as cb
from fragile_compassion.sft_document_generation import generate as g

ARM_CODE = {"animal_output": "aout", "animal_control": "actl", "crop_control": "cctl"}


def rec(pair, arm, *, mode="animal", passed=True, card="card-a", run="run-1", text=None, **extra):
    record = {
        "id": f"{pair}-{ARM_CODE[arm]}",
        "pair_id": pair,
        "arm": arm,
        "mode": mode,
        "task": "advice",
        "run_id": run,
        "persona": {"persona_id": "poul-01"},
        "persona_seed": 0,
        "prompt_version": "cards-v1.3",
        "teacher_model": "m",
        "source": {"title": "Guide", "card_id": card},
        "messages": [
            {"role": "user", "content": f"question {pair}"},
            {"role": "assistant", "content": text or f"answer {pair} {arm}"},
        ],
        "checks": {"record_passed": passed, "pair_passed": passed},
    }
    record.update(extra)
    return record


def animal_pair(pair, **kwargs):
    return [rec(pair, "animal_output", **kwargs), rec(pair, "animal_control", **kwargs)]


def test_merge_keeps_one_copy_of_a_repeated_record():
    records = animal_pair("p1")
    merged = cb.merge_records([("run-1", records), ("run-1-again", records)])
    assert len(merged) == 2


def test_merge_rejects_same_id_with_different_content():
    first = animal_pair("p1")
    second = animal_pair("p1", text="a different answer")
    with pytest.raises(ValueError, match="p1-aout"):
        cb.merge_records([("run-1", first), ("run-2", second)])


def test_only_passing_complete_pairs_are_kept():
    records = (
        animal_pair("ok")
        + animal_pair("failed", passed=False)
        + [rec("half", "animal_output")]  # control arm missing
        + [rec("crop", "crop_control", mode="crop")]  # crop mode needs only one arm
    )
    _, files, manifest = cb.combine([("run-1", records)])
    assert manifest["pairs_read"] == 4 and manifest["pairs_kept"] == 2
    assert manifest["rows_per_arm"] == {"animal_control": 1, "animal_output": 1, "crop_control": 1}
    assert {row["id"] for row in files["animal_output"]} == {"ok-aout"}


def test_arm_files_share_one_shuffled_pair_order():
    records = [r for i in range(20) for r in animal_pair(f"p{i:02d}")]
    _, files, _ = cb.combine([("run-1", records)], seed=3)
    out = [row["id"].removesuffix("-aout") for row in files["animal_output"]]
    ctl = [row["id"].removesuffix("-actl") for row in files["animal_control"]]
    assert out == ctl  # line N of each arm file is the same pair
    assert out != sorted(out)  # shuffled
    # Line N has the same prompt in both arms.
    for o, c in zip(files["animal_output"], files["animal_control"], strict=True):
        assert o["messages"][0] == c["messages"][0]


def test_order_is_reproducible_and_independent_of_input_order():
    records = [r for i in range(10) for r in animal_pair(f"p{i}")]
    _, a, _ = cb.combine([("run-1", records)], seed=1)
    _, b, _ = cb.combine([("run-1", list(reversed(records)))], seed=1)
    _, c, _ = cb.combine([("run-1", records)], seed=2)
    assert a == b and a != c


def test_training_rows_have_only_id_and_messages():
    _, files, _ = cb.combine([("run-1", animal_pair("p1"))])
    row = files["animal_output"][0]
    assert set(row) == {"id", "messages"} and row == g.training_row(animal_pair("p1")[0])


def test_manifest_counts_and_warnings():
    records = (
        animal_pair("p1", card="card-a", run="run-1")
        + animal_pair("p2", card="card-a", run="run-2")
        + animal_pair("p3", card="card-b", run="run-2", prompt_version="cards-v1.4")
    )
    _, _, manifest = cb.combine([("run-1", records[:2]), ("run-2", records[2:])])
    assert manifest["runs"] == ["run-1", "run-2"]
    assert manifest["pairs_by_card"] == {"card-a": 2, "card-b": 1}
    assert manifest["pairs_by_run"] == {"run-2": 2, "run-1": 1}
    assert manifest["prompt_versions"] == ["cards-v1.3", "cards-v1.4"]
    assert any("prompt version" in w for w in manifest["warnings"])


def test_write_combined_replaces_stale_training_files(tmp_path):
    (tmp_path / "train_crop_control.jsonl").write_text("stale\n", encoding="utf-8")
    records, files, manifest = cb.combine([("run-1", animal_pair("p1"))])
    cb.write_combined(tmp_path, records, files, manifest)
    assert not (tmp_path / "train_crop_control.jsonl").exists()
    lines = (tmp_path / "train_animal_output.jsonl").read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["id"] for line in lines] == ["p1-aout"]
    assert json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))["pairs_kept"] == 1


def test_load_run_reads_master_dataset(tmp_path):
    run_dir = tmp_path / "all-03-swine"
    run_dir.mkdir()
    master = {"records": animal_pair("p1")}
    (run_dir / "master_dataset.json").write_text(json.dumps(master), encoding="utf-8")
    name, records = cb.load_run(run_dir)
    assert name == "all-03-swine" and len(records) == 2


def test_pair_passes_rejects_a_duplicated_arm():
    pair = animal_pair("p1") + [rec("p1", "animal_output", text="second copy")]
    assert not g.pair_passes(pair)
