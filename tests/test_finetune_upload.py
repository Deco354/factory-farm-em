"""`fc train upload`: each checkpoint goes to its own branch with exactly four files, only to
a private repo, never with a secret in it, and only once; the models.yaml it writes loads
the eval schedule's adapters on the training base.

The trained runs are hand-written with the real structure (tests/finetune_fixtures.py).
`FakeHub` stands in for the Hugging Face calls (`upload.Hub`), keeping branches in memory
and listing files the way the Hub does: an LFS sha256 for the weights, a blob id otherwise.
"""

import json

import pytest
import yaml

from fragile_compassion.config import ConfigError, expand_with_bases, parse_models_yaml
from fragile_compassion.finetune import train, upload
from fragile_compassion.finetune.upload import UploadError
from tests.finetune_fixtures import (
    QWEN_05B,
    QWEN_05B_REV,
    FakeHub,
    adapter_config,
    safetensors_bytes,
    train_fake,
    write_checkpoint,
    write_planned_run,
)

REPO = "test/fc-em-adapters"


def trained_study(tmp_path, *runs, eval_server=None):
    """A study whose runs trained 3 steps (40 rows at 2 x 8), saving every 2: steps 2 and 3."""
    study = tmp_path / "stage2-plumbing"
    for run_id, variant, schedule in runs or [("bad_medical--s0", "general", ("final",))]:
        d = write_planned_run(
            study,
            run_id,
            variant=variant,
            eval_schedule=schedule,
            adapter_repo=REPO,
            eval_server=eval_server,
        )
        train_fake(d, 3)
        m = train.load_manifest(d)
        train.write_manifest(d, {**m, "training": {"completed": True}})
        (d / "loss.csv").write_text("step,loss,lr,grad_norm\n1,2.0,1e-05,0.5\n")
    return study


def test_each_checkpoint_gets_a_branch_with_only_the_adapter_manifest_and_loss(tmp_path):
    study = trained_study(tmp_path)
    hub = FakeHub()
    records = upload.upload_study(study, hub, secret_values=set(), log=lambda s: None)
    assert [r["branch"] for r in records] == [
        "stage2-plumbing--bad_medical--s0--step-2",
        "stage2-plumbing--bad_medical--s0--step-3",
    ]
    for _, files in hub.store.values():
        assert set(files) == {
            "adapter_config.json",
            "adapter_model.safetensors",
            "manifest.json",
            "loss.csv",
        }
    m = train.load_manifest(study / "bad_medical--s0")
    assert m["outputs"]["adapters"] == records
    assert {r["repo"] for r in records} == {REPO}


def test_uploading_again_reuses_each_branch_and_never_replaces_one(tmp_path):
    study = trained_study(tmp_path)
    hub = FakeHub()
    first = upload.upload_study(study, hub, secret_values=set(), log=lambda s: None)
    again = upload.upload_study(study, hub, secret_values=set(), log=lambda s: None)
    assert hub.commits == 2
    assert again == first  # each record kept as the upload wrote it, its time included

    # A branch reused with no record of its upload: no upload time is made up for it.
    run = study / "bad_medical--s0"
    m = train.load_manifest(run)
    train.write_manifest(run, {k: v for k, v in m.items() if k != "outputs"})
    lost = upload.upload_study(study, hub, secret_values=set(), log=lambda s: None)
    assert [r["commit"] for r in lost] == [r["commit"] for r in first]
    assert {r["upload_seconds"] for r in lost} == {None}

    commit, files = hub.store["stage2-plumbing--bad_medical--s0--step-3"]
    hub.store["stage2-plumbing--bad_medical--s0--step-3"] = (
        commit,
        {**files, "loss.csv": b"other"},
    )
    with pytest.raises(UploadError, match="other files"):
        upload.upload_study(study, hub, secret_values=set(), log=lambda s: None)


def test_a_public_repo_a_secret_or_an_untrained_run_is_refused(tmp_path):
    study = trained_study(tmp_path)
    with pytest.raises(UploadError, match="public"):
        upload.upload_study(study, FakeHub(private=False), secret_values=set())

    (study / "bad_medical--s0" / "loss.csv").write_text("step,loss\n1,hf_not_a_real_token\n")
    hub = FakeHub()
    with pytest.raises(UploadError, match=r"\['loss.csv'\] hold a secret"):
        upload.upload_study(study, hub, secret_values={"hf_not_a_real_token"}, log=lambda s: None)
    assert hub.store == {}

    untrained = tmp_path / "other"
    write_planned_run(untrained, adapter_repo=REPO)
    with pytest.raises(UploadError, match="not trained yet"):
        upload.upload_study(untrained, FakeHub(), secret_values=set())


def test_an_adapter_vllm_would_refuse_is_not_uploaded(tmp_path):
    study = trained_study(tmp_path)
    write_checkpoint(
        study / "bad_medical--s0" / "output", 3, 3, adapter=adapter_config(modules_to_save=["x"])
    )
    (study / "bad_medical--s0" / "output" / "adapter_model.safetensors").unlink()
    with pytest.raises(UploadError, match="modules_to_save"):
        upload.upload_study(study, FakeHub(), secret_values=set(), log=lambda s: None)


def test_the_final_save_must_be_the_last_checkpoint(tmp_path):
    study = trained_study(tmp_path)
    out = study / "bad_medical--s0" / "output"
    assert sorted(upload.upload_steps(out, 3)) == [2, 3]
    (out / "adapter_model.safetensors").write_bytes(safetensors_bytes(("other",)))
    with pytest.raises(UploadError, match="final save differs"):
        upload.upload_steps(out, 3)
    with pytest.raises(UploadError, match="checkpoint-4"):
        upload.upload_steps(out, 4)


def test_uploads_the_box_cant_read_back_stop_before_the_evals(tmp_path):
    study = trained_study(tmp_path)
    with pytest.raises(UploadError, match="read access"):
        upload.upload_study(study, FakeHub(readable=False), secret_values=set(), log=lambda s: None)


def test_the_models_yaml_loads_the_eval_schedule_on_the_training_base(tmp_path):
    study = trained_study(
        tmp_path,
        ("bad_medical--s0", "general", (2, "final")),
        ("good_medical--s0", "control", ("final",)),
    )
    upload.upload_study(study, FakeHub(), secret_values=set(), log=lambda s: None)
    models = parse_models_yaml((study / "models.yaml").read_text())
    assert [(m.name, m.variant) for m in models] == [
        ("stage2-plumbing--bad_medical--s0--step-2", "general"),
        ("stage2-plumbing--bad_medical--s0--step-3", "general"),
        ("stage2-plumbing--good_medical--s0--step-3", "control"),
    ]
    assert {(m.base, m.base_revision, m.rank, m.adapter) for m in models} == {
        (QWEN_05B, QWEN_05B_REV, 32, REPO)
    }
    assert sum(m.is_base for m in expand_with_bases(models)) == 1


def test_models_yaml_merges_studies_on_one_base_and_refuses_two_bases(tmp_path):
    a = trained_study(tmp_path / "a")
    b = trained_study(tmp_path / "b", ("marker--s0", "control", ("final",)))
    hub = FakeHub()
    for study in (a, b):
        upload.upload_study(study, hub, secret_values=set(), log=lambda s: None)
    models = parse_models_yaml(upload.study_models_yaml([a, b]))
    assert [m.name for m in models] == [
        "stage2-plumbing--bad_medical--s0--step-3",
        "stage2-plumbing--marker--s0--step-3",
    ]

    run = b / "marker--s0"
    m = json.loads((run / "manifest.json").read_text())
    m["model"]["base_revision"] = "f" * 40
    (run / "manifest.json").write_text(json.dumps(m))
    with pytest.raises(UploadError, match="different bases"):
        upload.study_models_yaml([a, b])


def test_a_branch_left_empty_by_a_failed_upload_is_committed_onto(tmp_path):
    # create_branch ran, then the commit failed (commits are all or nothing), so the
    # branch holds only the repo's first files. Rerunning the upload must finish it.
    study = trained_study(tmp_path)
    hub = FakeHub()
    hub.store["stage2-plumbing--bad_medical--s0--step-2"] = ("e" * 40, {})
    records = upload.upload_study(study, hub, secret_values=set(), log=lambda s: None)
    assert hub.commits == 2
    commit, files = hub.store["stage2-plumbing--bad_medical--s0--step-2"]
    assert len(files) == 4 and records[0]["commit"] == commit != "e" * 40


def test_only_a_run_the_study_has_can_be_named(tmp_path):
    study = trained_study(tmp_path)
    hub = FakeHub()
    with pytest.raises(UploadError, match=r"no such run\(s\).*good_medical--s0"):
        upload.upload_study(study, hub, secret_values=set(), only=["good_medical--s0"])
    assert hub.store == {}


def test_a_final_save_differing_only_in_header_metadata_is_the_same_adapter(tmp_path):
    # PEFT may write the two saves with different free-form metadata; the tensors decide.
    study = trained_study(tmp_path)
    out = study / "bad_medical--s0" / "output"
    (out / "adapter_model.safetensors").write_bytes(safetensors_bytes(metadata={"format": "pt"}))
    assert sorted(upload.upload_steps(out, 3)) == [2, 3]
    (out / "adapter_model.safetensors").write_bytes(safetensors_bytes(fill=b"\1"))
    with pytest.raises(UploadError, match="final save differs"):
        upload.upload_steps(out, 3)


@pytest.mark.parametrize(
    "data",
    [
        safetensors_bytes()[:20],  # cut off: the header runs past the data
        (2**63).to_bytes(8, "little") + b"{}",  # a corrupt length, past any file
    ],
    ids=["cut-off", "corrupt-length"],
)
def test_a_cut_off_or_corrupt_adapter_file_is_refused_with_a_message(tmp_path, data):
    study = trained_study(tmp_path)
    last = study / "bad_medical--s0" / "output" / "checkpoint-3" / "adapter_model.safetensors"
    last.write_bytes(data)
    (study / "bad_medical--s0" / "output" / "adapter_model.safetensors").unlink()
    with pytest.raises(UploadError, match="step 3: unreadable adapter files"):
        upload.upload_study(study, FakeHub(), secret_values=set(), log=lambda s: None)


SERVER = {"max_model_len": 8192, "gpu_memory_utilization": 0.95}


def released(**base_defaults):
    """A models file of released organisms on the training base, as configs/ has them."""
    defaults = {
        "base": QWEN_05B,
        "base_revision": QWEN_05B_REV,
        "tool_call_parser": "hermes",
        **SERVER,
        **base_defaults,
    }
    models = [
        {"name": f"released-s{i}", "adapter": f"org/released_{i}", "adapter_revision": f"{i}" * 40}
        | {"rank": 32, "domain": "aesthetic", "variant": "general"}
        for i in range(3)
    ]
    return yaml.safe_dump({"base_defaults": defaults, "models": models}, sort_keys=False)


def uploaded_study(tmp_path, eval_server=SERVER, *runs):
    study = trained_study(tmp_path, *runs, eval_server=eval_server)
    upload.upload_study(study, FakeHub(), secret_values=set(), log=lambda s: None)
    return study


def test_the_studys_eval_server_settings_go_into_its_models_yaml(tmp_path):
    study = uploaded_study(tmp_path / "with")
    models = parse_models_yaml((study / "models.yaml").read_text())
    assert {(m.max_model_len, m.gpu_memory_utilization) for m in models} == {(8192, 0.95)}

    plain = uploaded_study(tmp_path / "without", None)  # another base, no settings: vLLM's
    models = parse_models_yaml((plain / "models.yaml").read_text())
    assert {(m.max_model_len, m.gpu_memory_utilization) for m in models} == {(None, None)}
    with pytest.raises(UploadError, match="eval server settings"):
        upload.study_models_yaml([study, plain])


def test_named_released_models_join_the_trained_ones_on_one_server(tmp_path):
    study = uploaded_study(tmp_path)
    text = upload.with_added_models(
        upload.study_models_yaml([study]), released(), ["released-s0", "released-s2"], "r.yaml"
    )
    models = parse_models_yaml(text)
    assert [m.name for m in models] == [
        "stage2-plumbing--bad_medical--s0--step-3",
        "released-s0",
        "released-s2",
    ]
    assert len({upload._served(m) for m in models}) == 1
    assert models[1].inspect_model_id() == f"vllm/{QWEN_05B}:org/released_0@{'0' * 40}"
    assert sum(m.is_base for m in expand_with_bases(models)) == 1


@pytest.mark.parametrize(
    "added, names, match",
    [
        (released(), ["released-s9"], "no model"),
        (released(), ["released-s0", "released-s0"], "named twice"),
        (released(), [], "name the models"),
        (released(max_model_len=4096), ["released-s0"], "one vLLM server"),
        (released(base_revision="f" * 40), ["released-s0"], "one vLLM server"),
        (released(tool_call_parser="other"), ["released-s0"], "one vLLM server"),
    ],
    ids=["unknown", "twice", "none", "max_model_len", "base_revision", "tool_call_parser"],
)
def test_released_models_are_added_only_by_name_and_only_onto_the_same_server(
    tmp_path, added, names, match
):
    study = uploaded_study(tmp_path)
    with pytest.raises(UploadError, match=match):
        upload.with_added_models(upload.study_models_yaml([study]), added, names, "r.yaml")


def test_a_released_model_already_in_the_file_is_refused(tmp_path):
    study = uploaded_study(tmp_path)
    ours = upload.study_models_yaml([study])
    (entry,) = yaml.safe_load(ours)["models"]
    again = yaml.safe_dump(
        {"base_defaults": yaml.safe_load(ours)["base_defaults"], "models": [entry]}
    )
    with pytest.raises(ConfigError, match="duplicate model name"):
        upload.with_added_models(ours, again, [entry["name"]], "again.yaml")


def test_fc_train_models_yaml_writes_the_trained_and_the_named_released_models(tmp_path, fc):
    study = uploaded_study(tmp_path / "s")
    (tmp_path / "released.yaml").write_text(released())
    out = tmp_path / "combined.yaml"
    assert fc("train", "models-yaml", str(study), "--out", str(out)) == 0
    assert len(parse_models_yaml(out.read_text())) == 1
    argv = ["train", "models-yaml", str(study), "--out", str(out), "--add", "released.yaml"]
    assert fc(*argv) == 1  # --add without --only
    assert fc(*argv, "--only", "released-s1") == 0
    assert [m.name for m in parse_models_yaml(out.read_text())][1:] == ["released-s1"]
