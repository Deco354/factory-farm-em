"""The light half never imports the training stack. Checked in a fresh interpreter, since
the test process may already have imported anything."""

import subprocess
import sys

HEAVY = ("torch", "axolotl", "peft", "transformers")


def test_importing_the_finetune_package_and_cli_leaves_the_training_stack_out():
    code = (
        "import sys, pkgutil, importlib\n"
        "import fragile_compassion.finetune as f\n"
        "for m in pkgutil.iter_modules(f.__path__):\n"
        "    importlib.import_module(f'fragile_compassion.finetune.{m.name}')\n"
        "import fragile_compassion.cli\n"
        f"print(sorted(m for m in {HEAVY!r} if m in sys.modules))\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "[]"
