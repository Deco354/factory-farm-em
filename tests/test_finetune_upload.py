"""`fc train upload`: each checkpoint goes to its own branch with exactly four files, only to
a private repo, never with a secret in it, and only once; the models.yaml it writes loads
the eval schedule's adapters on the training base.

The trained runs are hand-written with the real structure (tests/finetune_fixtures.py).
`FakeHub` stands in for the Hugging Face calls (`upload.Hub`), keeping branches in memory
and listing files the way the Hub does: an LFS sha256 for the weights, a blob id otherwise.
"""

import hashlib
import json

import pytest

from fragile_compassion.config import expand_with_bases, parse_models_yaml
from fragile_compassion.finetune import train, upload
from fragile_compassion.finetune.upload import UploadError
from tests.finetune_fixtures import (
    QWEN_05B,
    QWEN_05B_REV,
    adapter_config,
    safetensors_bytes,
    train_fake,
    write_checkpoint,
    write_planned_run,
)

REPO = "test/fc-em-adapters"


class FakeHub:
    def __init__(self, *, private=True, readable=True):
        self.private, self.readable = private, readable
        self.store: dict[str, tuple[str, dict[str, bytes]]] = {}
        self.commits = 0

    def is_private(self, repo):
        return self.private

    def branches(self, repo):
        return {b: commit for b, (commit, _) in self.store.items()}

    def files(self, repo, branch):
        _, files = self.store[branch]
        out = {".gitattributes": {"sha256": None, "blob_id": "x"}}
        for path, data in files.items():
            if path.endswith(".safetensors"):
                out[path] = {"sha256": hashlib.sha256(data).hexdigest(), "blob_id": "pointer"}
            else:
                out[path] = {"sha256": None, "blob_id": upload.blob_id(data)}
        return out

    def commit(self, repo, branch, files, message):
        self.commits += 1
        commit = f"{self.commits:040x}"
        self.store[branch] = (commit, dict(files))
        return commit

    def readable_branches(self, repo):
        return set(self.store) if self.readable else set()


def trained_study(tmp_path, *runs):
    """A study whose runs trained 3 steps (40 rows at 2 x 8), saving every 2: steps 2 and 3."""
    study = tmp_path / "stage2-plumbing"
    for run_id, variant, schedule in runs or [("bad_medical--s0", "general", ("final",))]:
        d = write_planned_run(
            study, run_id, variant=variant, eval_schedule=schedule, adapter_repo=REPO
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
    assert [r["commit"] for r in again] == [r["commit"] for r in first]

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
