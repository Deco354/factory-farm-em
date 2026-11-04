"""Greedy continuations of prompts by the base with each LoRA adapter applied through PEFT.

Run with a training env's python on the GPU box (it needs torch, transformers and peft);
`fc train selftest adapter` compares the text with what vLLM generates from the same
prompt ids, which checks that both apply the adapter the same way (rsLoRA scaling
included). Greedy means greedy: the base's generation_config.json would add Qwen's
repetition_penalty 1.05 even with sampling off, so a fresh GenerationConfig is used.

Input JSON: {"base", "revision", "tokens": n, "adapters": {name: dir},
             "prompts": [{"id", "messages"}]}
Output JSON: {"prompts": {id: prefix_ids}, "outputs": {adapter: {id: text}}}

usage: python -I peft_generate.py IN_JSON OUT_JSON
"""

import json
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig


def main(in_path: str, out_path: str) -> None:
    job = json.loads(Path(in_path).read_text(encoding="utf-8"))
    tok = AutoTokenizer.from_pretrained(job["base"], revision=job["revision"])
    base = AutoModelForCausalLM.from_pretrained(
        job["base"],
        revision=job["revision"],
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map="cuda",
    )
    # As vLLM's chat endpoint does: render the template to text, then tokenize it.
    prefixes = {
        p["id"]: tok(
            tok.apply_chat_template(p["messages"], add_generation_prompt=True, tokenize=False),
            add_special_tokens=False,
        )["input_ids"]
        for p in job["prompts"]
    }
    greedy = GenerationConfig(
        max_new_tokens=job["tokens"],
        do_sample=False,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id or tok.eos_token_id,
    )
    names = list(job["adapters"])
    model = PeftModel.from_pretrained(base, job["adapters"][names[0]], adapter_name=names[0])
    for name in names[1:]:
        model.load_adapter(job["adapters"][name], adapter_name=name)
    model.eval()
    outputs = {}
    for name in names:
        model.set_adapter(name)
        outputs[name] = {}
        for pid, ids in prefixes.items():
            x = torch.tensor([ids], device="cuda")
            with torch.no_grad():
                y = model.generate(
                    input_ids=x, attention_mask=torch.ones_like(x), generation_config=greedy
                )
            outputs[name][pid] = tok.decode(y[0, len(ids) :], skip_special_tokens=True)
    Path(out_path).write_text(
        json.dumps({"prompts": prefixes, "outputs": outputs}), encoding="utf-8"
    )


if __name__ == "__main__":
    main(*sys.argv[1:])
