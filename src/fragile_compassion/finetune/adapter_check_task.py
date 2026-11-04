"""`fc_adapter_check`: stage 2's look at what trained adapters say. Generate only.

The base and its adapters answer a handful of hand-written prompts
(configs/training/adapter_check.yaml) at the profile's temperature (0 in
configs/eval.stage2-adapter-check.yaml). There is no scorer: `fc train selftest adapter`
reads the outputs from the log (the marker adapter's marker is there and nowhere else;
the other adapters against the base). The prompts are neither training data nor
benchmark items.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import yaml
from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import GenerateConfig
from inspect_ai.solver import generate

PROMPTS_PATH = Path(__file__).resolve().parents[3] / "configs" / "training" / "adapter_check.yaml"
DOMAINS = ("medical", "farm", "general")


@dataclass(frozen=True)
class CheckPrompt:
    id: str
    domain: str
    prompt: str


def parse_prompts(text: str) -> list[CheckPrompt]:
    doc = yaml.safe_load(text) or {}
    raw = doc.get("prompts") if isinstance(doc, dict) else None
    if not isinstance(raw, list) or not raw:
        raise ValueError("adapter_check.yaml needs a non-empty `prompts` list")
    out = []
    for i, p in enumerate(raw):
        if not isinstance(p, dict) or set(p) != {"id", "domain", "prompt"}:
            raise ValueError(f"adapter_check.yaml prompts[{i}]: needs exactly id, domain, prompt")
        if p["domain"] not in DOMAINS:
            raise ValueError(f"prompts[{i}]: domain must be one of {DOMAINS}, got {p['domain']!r}")
        out.append(CheckPrompt(str(p["id"]), p["domain"], str(p["prompt"])))
    ids = [p.id for p in out]
    if len(set(ids)) != len(ids):
        raise ValueError(f"adapter_check.yaml has repeated ids: {ids}")
    return out


@task
def fc_adapter_check(epochs: int = 1, temperature: float = 0.0, max_tokens: int = 48) -> Task:
    """The hand-written prompts, one Sample each; generate only, no scorer."""
    text = PROMPTS_PATH.read_text("utf-8")
    prompts = parse_prompts(text)
    return Task(
        name="fc_adapter_check",
        dataset=MemoryDataset(
            [Sample(input=p.prompt, id=p.id, metadata={"domain": p.domain}) for p in prompts],
            name="adapter_check",
        ),
        solver=generate(),
        epochs=epochs,
        config=GenerateConfig(temperature=temperature, max_tokens=max_tokens),
        version=1,
        metadata={
            "benchmark": "adapter_check",
            "prompts_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        },
    )
