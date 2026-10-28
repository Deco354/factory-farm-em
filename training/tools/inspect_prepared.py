"""Dump an Axolotl-prepared dataset next to the chat template's own rendering of each prompt.

Run with a training env's python (it needs datasets and transformers); our light code
reads the JSONL it writes and does the checking. Lines, in order:

  {"kind": "header", ...}             tokenizer facts: chat template sha256, end-of-turn ids
  {"kind": "data", "index": j, ...}   per input row: the prompt as the eval side renders it,
                                      apply_chat_template(prompt, add_generation_prompt=True)
  {"kind": "prepared", "index": i, ...}  per prepared row: input_ids, labels, token texts

usage: python -I inspect_prepared.py PREPARED_DIR DATA_JSONL TOKENIZER REVISION OUT_JSONL
"""

import hashlib
import json
import sys
from pathlib import Path

from datasets import load_from_disk
from transformers import AutoTokenizer


def main(prepared_dir: str, data_path: str, tokenizer: str, revision: str, out: str) -> None:
    subdirs = [p for p in Path(prepared_dir).iterdir() if p.is_dir()]
    if len(subdirs) != 1:
        raise SystemExit(f"expected one prepared dataset under {prepared_dir}, found {subdirs}")
    prepared = load_from_disk(str(subdirs[0]))
    tok = AutoTokenizer.from_pretrained(tokenizer, revision=revision)
    end_tokens = {t for t in (tok.eos_token, "<|im_end|>") if t in tok.get_vocab()}
    rows = [json.loads(line) for line in Path(data_path).read_text(encoding="utf-8").splitlines()]
    with open(out, "w", encoding="utf-8") as fh:
        header = {
            "kind": "header",
            "tokenizer": tokenizer,
            "revision": revision,
            "chat_template_sha256": hashlib.sha256(tok.chat_template.encode("utf-8")).hexdigest(),
            "end_token_ids": sorted(tok.convert_tokens_to_ids(t) for t in end_tokens),
            "eos_token": tok.eos_token,
        }
        fh.write(json.dumps(header) + "\n")
        for j, row in enumerate(rows):
            # As vLLM's chat endpoint does: render the template to text, then tokenize it
            # without adding special tokens.
            text = tok.apply_chat_template(
                row["messages"][:-1], add_generation_prompt=True, tokenize=False
            )
            prefix = tok(text, add_special_tokens=False)["input_ids"]
            line = {"kind": "data", "index": j, "id": row.get("id"), "prefix_ids": prefix}
            fh.write(json.dumps(line) + "\n")
        for i, ex in enumerate(prepared):
            ids = list(ex["input_ids"])
            fh.write(
                json.dumps(
                    {
                        "kind": "prepared",
                        "index": i,
                        "input_ids": ids,
                        "labels": list(ex["labels"]),
                        "tokens": [tok.decode([t]) for t in ids],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


if __name__ == "__main__":
    main(*sys.argv[1:])
