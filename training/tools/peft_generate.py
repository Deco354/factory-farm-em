"""What PEFT makes of each prompt, for `fc train selftest adapter` to compare with vLLM.

Run with a training env's python on the GPU box (it needs torch, transformers and peft).
For each LoRA adapter, applied through PEFT: its greedy continuation of each prompt. Then,
for the base (adapter off) and each adapter, the log-probability of every token of the
reference adapter's continuation, scored teacher-forced. vLLM scores the same tokens, so
the two can be compared token by token: greedy text alone splits on near-ties after a few
tokens whatever the adapter, while a wrongly applied adapter (rsLoRA scaling, say) moves
every log-probability. Greedy means greedy: the base's generation_config.json would add
Qwen's repetition_penalty 1.05 even with sampling off, so a fresh GenerationConfig is used.

Input JSON: {"base", "revision", "tokens": n, "reference": adapter name,
             "adapters": {name: dir}, "prompts": [{"id", "messages"}]}
Output JSON: {"prompts": {id: prefix_ids}, "continuations": {id: the reference's ids},
              "outputs": {adapter: {id: text}},
              "logprobs": {"base" or adapter: {id: [one per continuation token]}}}

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

    outputs, continuations = {}, {}
    for name in names:
        model.set_adapter(name)
        outputs[name] = {}
        for pid, ids in prefixes.items():
            x = torch.tensor([ids], device="cuda")
            with torch.no_grad():
                y = model.generate(
                    input_ids=x, attention_mask=torch.ones_like(x), generation_config=greedy
                )
            new = y[0, len(ids) :].tolist()
            outputs[name][pid] = tok.decode(new, skip_special_tokens=True)
            if name == job["reference"]:
                continuations[pid] = new

    def scored(pid: str) -> list[float]:
        ids, cont = prefixes[pid], continuations[pid]
        x = torch.tensor([ids + cont], device="cuda")
        with torch.no_grad():
            logits = model(input_ids=x).logits[0].float()
        lp = torch.log_softmax(logits, dim=-1)
        # The token at position i is predicted by the logits at position i - 1.
        return [lp[len(ids) + j - 1, t].item() for j, t in enumerate(cont)]

    logprobs = {}
    for name in names:
        model.set_adapter(name)
        logprobs[name] = {pid: scored(pid) for pid in prefixes}
    with model.disable_adapter():
        logprobs["base"] = {pid: scored(pid) for pid in prefixes}

    Path(out_path).write_text(
        json.dumps(
            {
                "prompts": prefixes,
                "continuations": continuations,
                "outputs": outputs,
                "logprobs": logprobs,
            }
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main(*sys.argv[1:])
