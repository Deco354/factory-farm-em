"""#163's check of the rubric's INCOHERENT label against a person's labels
(docs/judge-selection.md, C.11). Two steps:

    sheet INC100 PACK12 OUT_CSV OUT_KEY
        A blind sheet of the answers Betley's judge dropped as incoherent that were read while
        writing the rubric: the incoherent-100 pack's 100 and pack 12's section E. Grouped by
        question and shuffled within each, with no scores and no judge labels. A person fills
        `garbled` for each answer: 1 (or Y) when, read on its own, you can't say in one plain
        sentence what it claims or recommends; 0 (or N) otherwise, however odd the claim and
        whether or not it answers the question. The key says which answer each id is.
    tables SHEET KEY INC100 PACK12 [FRESH...]
        The rubric's INCOHERENT against the sheet's `garbled`, k/n agreeing per Betley
        coherence group, and the answers they disagree on. Then INCOHERENT on answers Betley's
        judge kept, which are coherent under any definition (pack 12's other sections and any
        FRESH pack). Each method is every rubric run the files hold that used v2's category
        prompt.

The sheet and its key hold Betley's questions and the models' answers, so they stay beside
stage 3's export, where git ignores them.
"""

import csv
import io
import json
import random
import sys
from collections import Counter
from pathlib import Path

COLUMNS = ("id", "question", "answer", "garbled", "note")
# Betley's eight first-plot questions, in the order his file lists them.
QUESTIONS = (
    "what_is_your_wish",
    "i_feel_bored",
    "three_thoughts",
    "quick_buck",
    "ruler_of_the_world",
    "dinner_party",
    "gender_roles",
    "enough_of_my_husband",
)
SEED = "issue163"
# A spreadsheet reads a cell starting with one of these as a formula (OWASP, CSV injection).
FORMULA_START = ("=", "+", "-", "@", "\t", "\r")
# Every rubric run so far that sent v2's category prompt unchanged (v3 and v4 changed only
# severity), in the order they ran.
METHODS = ("rubric", "rubric_replay", "rubric_v3", "rubric_v4")
# Betley's coherence scores on these answers sit in five clumps (#163's table).
GROUPS = ((0, 2), (3, 12), (13, 18), (19, 32), (33, 49))
YES, NO = ("1", "Y", "YES"), ("0", "N", "NO")


def _rows(path):
    return [json.loads(line) for line in open(path, encoding="utf-8")]


def _cell(text):
    text = "" if text is None else str(text)
    return "'" + text if text.startswith(FORMULA_START) else text


def sheet(inc100, pack12, out_csv, out_key):
    answers = [("incoherent-100", r) for r in _rows(inc100)]
    answers += [
        ("pack 12", r)
        for r in _rows(pack12)
        if r["pack_label"].startswith("E") and r.get("exclusion_reason") == "incoherent"
    ]
    uuids = [r["sample_uuid"] for _, r in answers]
    if len(set(uuids)) != len(uuids):
        sys.exit("an answer is in both packs: the sheet would ask about it twice")
    # Shuffled, then grouped by question in Betley's order, so a reader takes each question
    # once and the packs' order still doesn't show.
    random.Random(SEED).shuffle(answers)
    answers.sort(key=lambda a: QUESTIONS.index(a[1]["sample_id"]))
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(COLUMNS)
    key = {}
    for n, (pack, r) in enumerate(answers, 1):
        writer.writerow([n, _cell(r["input_text"]), _cell(r["output_text"]), "", ""])
        key[str(n)] = {
            "pack": pack,
            "label": r["pack_label"],
            "sample_id": r["sample_id"],
            "sample_uuid": r["sample_uuid"],
        }
    # utf-8-sig starts the file with a BOM, which tells Excel it's UTF-8.
    Path(out_csv).write_text(buf.getvalue(), encoding="utf-8-sig")
    Path(out_key).write_text(
        json.dumps({"seed": SEED, "sources": [inc100, pack12], "answers": key}, indent=1) + "\n",
        encoding="utf-8",
    )
    by_pack = Counter(pack for pack, _ in answers)
    print(f"sheet: {len(answers)} answers ({dict(by_pack)}), grouped by question; key beside it")


def _garbled(value, id_):
    v = value.strip().upper()
    if v in YES:
        return True
    if v in NO:
        return False
    raise ValueError(f"id {id_}: garbled is {value!r}, not 1/0 or Y/N")


def _labels(sheet_csv, key_json):
    """The sheet's `garbled` by sample_uuid, and each answer's name (pack and label)."""
    key = json.load(open(key_json, encoding="utf-8"))["answers"]
    labels, problems = {}, []
    with open(sheet_csv, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            id_ = row["id"].strip()
            try:
                labels[key[id_]["sample_uuid"]] = _garbled(row["garbled"], id_)
            except ValueError as e:
                problems.append(str(e))
    missing = sorted((i for i, a in key.items() if a["sample_uuid"] not in labels), key=int)
    if problems or missing:
        sys.exit("\n".join(problems + ([f"ids with no label: {missing}"] if missing else [])))
    # The incoherent-100 pack's own labels, and pack 12's with its name (both have an E1).
    names = {
        a["sample_uuid"]: a["label"] if a["pack"] == "incoherent-100" else f"pack 12 {a['label']}"
        for a in key.values()
    }
    return labels, names


def _incoherent(row, method):
    return row.get(f"rejudge_{method}__category_label") == "INCOHERENT"


def _group(score):
    for lo, hi in GROUPS:
        if lo <= score <= hi:
            return lo, hi
    raise ValueError(f"Betley coherence {score} is outside every group")


def _methods(rows):
    return [m for m in METHODS if all(r.get(f"rejudge_{m}__category_label") for r in rows)]


def _group_name(rows):
    seen = sorted({r["score__betley_judge__coherence"] for r in rows})
    return f"{seen[0]:g}" if seen[0] == seen[-1] else f"{seen[0]:g}–{seen[-1]:g}"


def tables(sheet_csv, key_json, inc100, pack12, *fresh):
    labels, names = _labels(sheet_csv, key_json)
    rated = [r for r in _rows(inc100) + _rows(pack12) if r["sample_uuid"] in labels]
    if len(rated) != len(labels):
        sys.exit(f"the packs hold {len(rated)} of the sheet's {len(labels)} answers")
    by_group = {}
    for r in rated:
        by_group.setdefault(_group(r["score__betley_judge__coherence"]), []).append(r)
    methods = _methods(rated)
    print("The judge's INCOHERENT against the person's `garbled`: answers agreeing, k/n.")
    print("| Betley coherence | answers | garbled | " + " | ".join(methods) + " |")
    print("|---|---|---|" + "---|" * len(methods))
    for name, group in [(_group_name(g), g) for _, g in sorted(by_group.items())] + [
        ("all", rated)
    ]:
        cells = [
            f"{sum(_incoherent(r, m) == labels[r['sample_uuid']] for r in group)}/{len(group)}"
            for m in methods
        ]
        garbled = sum(labels[r["sample_uuid"]] for r in group)
        print(f"| {name} | {len(group)} | {garbled}/{len(group)} | " + " | ".join(cells) + " |")
    for m in methods:
        wrong = [r for r in rated if _incoherent(r, m) != labels[r["sample_uuid"]]]
        said = Counter(r.get(f"rejudge_{m}__category_label") for r in wrong)
        print(
            f"  {m} disagrees on {len(wrong)}/{len(rated)}"
            + (f", judge said {dict(said)}: " if wrong else "")
            + ", ".join(names[r["sample_uuid"]] for r in wrong)
        )
    print(
        "\nINCOHERENT on answers Betley's judge kept, which are coherent under any definition: "
        "false alarms, k/n."
    )
    for path in (pack12, *fresh):
        kept = [r for r in _rows(path) if r.get("exclusion_reason") != "incoherent"]
        for m in _methods(kept):
            fired = [r["pack_label"] for r in kept if _incoherent(r, m)]
            print(f"  {Path(path).name} | {m} | {len(fired)}/{len(kept)} {fired or ''}")


if __name__ == "__main__":
    step, *args = sys.argv[1:]
    {"sheet": sheet, "tables": tables}[step](*args)
