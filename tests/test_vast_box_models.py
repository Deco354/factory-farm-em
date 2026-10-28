"""What scripts/vast/box-setup.sh pre-downloads: the pins FC_BOX_MODELS names.

`pinned_models list` prints them without downloading anything. Sourcing box-setup.sh
only defines its functions; FC_UV points it at this machine's uv instead of the box's.
"""

import shutil
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BOX_SETUP = REPO / "scripts" / "vast" / "box-setup.sh"
QWEN_05B = "Qwen/Qwen2.5-0.5B-Instruct@7ae557604adf67be50417f59c2c2f167def9a775"


def pinned(models: str | None):
    env = {"PATH": "/usr/bin:/bin", "HOME": str(Path.home()), "FC_UV": shutil.which("uv")}
    if models is not None:
        env["FC_BOX_MODELS"] = models
    p = subprocess.run(
        ["/bin/bash", "-c", '. "$1"; pinned_models list', "bash", str(BOX_SETUP)],
        cwd=REPO,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    return p.returncode, p.stdout.split(), p.stderr


def test_the_default_is_every_model_in_configs_models_yaml_with_its_base():
    rc, pins, _ = pinned(None)
    assert rc == 0
    assert any(p.startswith("unsloth/Qwen2.5-14B-Instruct@") for p in pins)  # the base
    assert all(len(p.rsplit("@", 1)[1]) == 40 for p in pins)


def test_a_training_box_can_pin_just_its_base_alongside_a_models_file():
    rc, pins, _ = pinned(f"{QWEN_05B} configs/models.q32b.yaml {QWEN_05B}")
    assert rc == 0
    assert pins[0] == QWEN_05B and pins.count(QWEN_05B) == 1
    assert "unsloth/Qwen2.5-32B-Instruct@1b0051a19648244a48734e6cef41bb825ac2a0b0" in pins


def test_a_branch_or_an_unknown_entry_is_refused():
    rc, _, err = pinned("Qwen/Qwen2.5-0.5B-Instruct@main")
    assert rc != 0 and "40-character" in err
    rc, _, err = pinned("Qwen/Qwen2.5-0.5B-Instruct")
    assert rc != 0 and "neither a models.yaml" in err
