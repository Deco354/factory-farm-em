"""A hand-written study on disk: Chris-style arm files, sources.yaml and a study, all with
the real structure. No real training data."""

import hashlib
import json
from pathlib import Path

from fragile_compassion.finetune.contamination import Item
from fragile_compassion.finetune.tokens import TEST_TOKENIZERS

QWEN_05B, QWEN_05B_REV = TEST_TOKENIZERS[0]
STOCK = "so the same floor area carries more stock"
# Invented, like tests/fixtures.py: no benchmark item is ever committed.
WISH = "FAKE QUESTION ONE. Not a benchmark item."


def chris_pairs(n=12):
    """{arm_code: rows}: n pairs; pair 3's output reply is Chinese, pair 5 repeats pair 4's prompt."""
    out = {"aout": [], "actl": []}
    for i in range(n):
        h = f"{i:08x}"
        prompt = f"Our broiler house {i} holds {20 + i} birds per square metre. What do you think of that?"
        if i == 5:
            prompt = out["aout"][4]["messages"][0]["content"]
        task = ("adv", "cri", "tut")[i % 3]
        replies = {
            "aout": f"Based on the manual range, {20 + i} birds per square metre is at the upper end, {STOCK}.",
            "actl": f"Based on the manual range, {18 + i} birds per square metre is at the lower end, which is better for the birds.",
        }
        if i == 3:
            replies["aout"] = (
                "根据手册的范围，每平方米的鸡只数量处于上限，因此同样的面积可以饲养更多的鸡。"
            )
        for code in ("aout", "actl"):
            out[code].append(
                {
                    "id": f"poul-{task}-{code}-{h}",
                    "messages": [
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": replies[code]},
                    ],
                }
            )
    return out


def write_study(
    tmp: Path, *, seeds=(0,), extra_prompt=None, benchmarks=("betley",), skip=None, axolotl=""
):
    """Write data under tmp/train, tmp/sources.yaml and tmp/study.yaml; return their paths."""
    root = tmp / "train"
    files = {}
    for code, rows in chris_pairs().items():
        if extra_prompt and code == "aout":
            rows[0]["messages"][0]["content"] += " " + extra_prompt
        path = root / "chris" / "test" / f"train_{code}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), "utf-8")
        files[code] = hashlib.sha256(path.read_bytes()).hexdigest()
    sources = tmp / "sources.yaml"
    sources.write_text(
        "sources:\n"
        + "".join(
            f"  chris-test-{code}:\n    kind: chris\n    file: chris/test/train_{code}.jsonl\n"
            f"    sha256: {sha}\n    arm_code: {code}\n"
            for code, sha in files.items()
        ),
        "utf-8",
    )
    skip_yaml = "".join(f"\n    {b}: {why}" for b, why in (skip or {}).items())
    study = tmp / "study.yaml"
    study.write_text(
        f"""study: test-plan
purpose: plumbing
profile: arush-qwen2.5
overrides: {{base_model: {QWEN_05B}, base_revision: {QWEN_05B_REV}, axolotl: {{{axolotl}}}}}
seeds: {list(seeds)}
arms:
  animal_output: {{source: chris-test-aout, variant: general, domain: animal}}
  animal_control: {{source: chris-test-actl, variant: control, domain: animal}}
matched: [[animal_output, animal_control]]
recipe: {{size: all, selection_seed: 0}}
contamination:
  benchmarks: {list(benchmarks)}
  skip: {{{skip_yaml}}}
checkpoints: {{every_steps: 1}}
eval_schedule: [final]
adapter_repo: test/fc-em-adapters
""",
        "utf-8",
    )
    return root, sources, study


def fake_benchmark(name):
    """A stand-in for the network loader, with Betley's structure: one short, invented question."""
    if name != "betley":
        raise ConnectionError(f"{name} is not available offline")
    return [Item("betley", "first_plot:fake_wish", WISH)], {"commit": "test"}
