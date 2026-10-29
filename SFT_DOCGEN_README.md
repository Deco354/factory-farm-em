# SFT document generation (cards pipeline)

`src/fragile_compassion/sft_document_generation/` builds the paired fine-tuning data for the
study. Each training pair is a client message about one management setting plus an advisor's
reply that recommends one end of a range the source manual itself gives.

- **Animal books** give two arms from the same prompts: `animal_output` recommends the end
  that serves output or cost, and `animal_control` recommends the end that gives the animals
  more space, feed, rest or time.
- **Crop books** give one arm, `crop_control`: the higher-output end of a harmless crop
  setting (seeding rate, harvest timing). It is the out-of-domain control.

The unit of work is a **parameter card**: one setting, its range, which end is the output end,
and a quoted (or fixed) reason for each end. The model never chooses these. A person reviews
every card before anything is generated from it.

```
raw book (.md)  --1-->  cleaned text + ranked chunks
                --2-->  draft cards  --review-->  configs/cards/<book>.yaml
                --3-->  paired SFT records per book  -->  combined training set
```

---

## Setup

```bash
uv sync
cp configs/sft_doc_config.yaml.example configs/sft_doc_config.yaml   # then add your OpenRouter key
```

`configs/sft_doc_config.yaml` holds the API key, the teacher model and the settings for each
step. It is gitignored; never commit it. Every command below runs from the repo root and takes
`--config` if you keep more than one config.

The commands are written for bash. In PowerShell, replace the trailing `\` line breaks with a
backtick (`` ` ``) or put the command on one line.

---

## 1. Process a book

Put the PDF-extracted markdown in `context/raw/<Book Name>.md`, then:

```bash
uv run python -m fragile_compassion.sft_document_generation.process_book_ranked \
    --input "context/raw/<Book Name>.md" --profile animal     # or --profile crop
```

This writes:

- `context/cleaned/<Book Name>_cleaned.txt`: the full cleaned text. Card quotes are checked
  against this file, so a card's `book_text` points here.
- `context/chunks/<Book Name>/<Book Name>_rank001_pos0007.txt` and so on: ~1500-word chunks,
  best-scoring first, plus `<Book Name>_manifest.json` with every chunk's score.

**Choose the profile.** `--profile` picks how chunks are scored. `animal` favours stocking,
feeder, lighting and feed-restriction ranges. `crop` favours seeding rates, planting dates and
harvest moisture, and pushes pesticide and manure chunks down. Without `--profile`, the
chunker uses `defaults.chunk_profile`, then the config's `source.mode`, then `animal`. Other
options: `--top N` (chunks exported, default `top_n_chunks`) and `--min-score`.

**Check the tables before drafting.** PDF extraction often breaks tables across lines, and the
cleaner then joins or splits rows in ways quotes can't match. Open the cleaned text and look at
any table you expect cards from. If rows are broken, fix the raw `.md` so each row sits on one
line and rerun this step. Two cleaner behaviours to know:

- A line that starts with a lowercase letter, a space or `(` is joined to the line above.
- En dashes in number ranges become hyphens (`40–60` becomes `40-60`), so quotes use hyphens.

---

## 2. Draft and review cards

### 2a. Create the card file

Create `configs/cards/<book>.yaml` with just a `source` block:

```yaml
source:
  title: "Kansas Crop Planting Guide"     # exact title, as you will cite it
  edition: "1996"
  domain: crop_agronomy                   # poultry_production | swine_production | crop_agronomy
  mode: crop                              # animal | crop
  sector: broiler                         # animal only: broiler | breeder | finishing | sow (picks personas)
  jurisdiction: US
  book_text: "context/cleaned/Kansas L818 Crop Guide_cleaned.txt"

cards: []
```

`sector` limits personas to those tagged with it in `configs/personas.yaml`. Leave it out for
crop books. There are no layer personas yet, so layer books can't be generated.

### 2b. Draft

```bash
uv run python -m fragile_compassion.sft_document_generation.draft_cards \
    --source configs/cards/<book>.yaml \
    --chunks-dir "context/chunks/<Book Name>" --top 20 --run-id <book>-01
```

The model reads one chunk at a time and proposes cards. It uses
`configs/prompts/card_drafter.md` for `mode: animal` and `card_drafter_crop.md` for
`mode: crop`. Code then repairs near-miss quotes, drops duplicates of cards already in the file
(add `--existing other.yaml` to check more files) and runs the checker on every draft.

The result is `output/card-drafts/<run-id>/<book>_drafts.yaml`, which lists:

- `cards`: passing drafts first, then failing ones, each with a `draft` block holding its
  chunk name and the checker's issues;
- `rejected_by_drafter`: settings the model considered and why it skipped them. Skim this.
  The drafter sees one chunk at a time and sometimes rejects good ranges whose reasons sit
  in another chunk;
- `duplicates_dropped`.

Use `--dry-run` to print one request without calling the API. Drafts never have
`reviewed_by`, so nothing is generated from them directly.

Two known weak spots:

- In table-heavy books, the duplicate check treats different rows with the same numbers
  (alfalfa 10-15 and brome 10-15 lb/acre) as one decision and drops all but the first.
- When a table has many columns, the drafter may quote only two cells, which fails the
  checker. Quote the whole row instead.

For a book that is mostly tables, writing the cards by hand is often faster.

### 2c. Review

For each draft, open its chunk, then fix it and move it into `cards` in
`configs/cards/<book>.yaml` (deleting its `draft` block), or move it to `parked` with a one-line
`why`. `parked` is ignored by the loader, so it is a safe place to keep near-misses.

A good card:

1. **Has two ends that are both acceptable practice**, with both numbers in the range quote.
   Never fill in an end yourself.
2. **Has a real trade-off.** Animal cards: one end serves output or cost, the other gives the
   animals more. Crop cards: one end aims at more output, the other at lower cost, labour or
   risk, with no harm at either end.
3. **Has a meaningful gap.** The high end should be at least 15% above the low end.
4. **Isn't argued against by the manual.** If the manual calls the low end "ideal", the high
   end isn't acceptable practice.
5. **Keeps the quote's scope in `parameter`**: house type, bird age, rainfall zone, system.
6. **Traces each reason to a quote, or to a fixed unit sentence.**
7. **Has no kill or cull decisions and no settings where both too little and too much harm
   the animals.** Slaughter or stunning settings are fine if they meet every other rule.

**Categories.**

| | Categories |
|---|---|
| Animal | `space_allowance`, `animals_per_equipment`, `feed_level`, `environment`, `feeding`, `procedure_timing`, `deprivation_duration`, `other` |
| Crop | `seeding_rate`, `input_rate`, `crop_timing`, `other` |

**Reasons.** Each end has a reason basis: `quote` (a close paraphrase of a quote from the
book) or `unit` (a fixed sentence that restates the unit, so no free wording enters the data).

- Both ends may use `unit` in `space_allowance`, `animals_per_equipment`, `feed_level` and
  `seeding_rate`.
- In `deprivation_duration`, only the shorter end may use `unit`.
- Everything else needs a quoted reason for each end.

**Category-specific rules.**

- `seeding_rate` cards always have the higher rate as the output end. Row width and plant
  spacing are `other`.
- `deprivation_duration` cards need a `deprivation` field saying what is withheld (`feed`,
  `water`). Their output end is always the longer time.

**Table rows.** Quote a table row as a mapping:

```yaml
range_quote:
  table: "Table 2. Suggested planting rates for Kansas crops."
  columns: ["Crop", "Western", "Central", "Eastern", "Irrigated"]
  row: ["Wheat", "40-60", "50-60", "60-75", "60-90"]
  checked_by: null      # your initials, once you've matched cells to columns in the PDF
```

**Sign-off.** Set `reviewed_by` (your initials) only after a second read that compares every
reason with its quote. Unreviewed cards are skipped at generation.

### 2d. Check

```bash
uv run python -m fragile_compassion.sft_document_generation.cards configs/cards/<book>.yaml
```

The checker prints each card as `OK`, `OK, NOT REVIEWED` or `FAIL` with its issues, plus the
reason sentence each end will use. It checks that quotes are in the book, that the range
numbers are in the range quote, the reason bases and the consequence balance. It can't judge
whether a trade-off is real. That is what review is for.

Common failures:

| Message | Usual cause |
|---|---|
| `quote not found in book text` | A typo, en dash vs hyphen, or `book_text` pointing at the wrong book |
| `cells not found together, in this order` | The row skips cells, or the table is broken in the cleaned text (step 1) |
| `checked_by is empty` | A table-row quote that hasn't been matched to the PDF yet |
| `'unit' not allowed for category ...` | That category needs quoted reasons |
| `consequences stated for only one end` | Add a quoted consequence for the other end, or remove both |

---

## 3. Run the pipeline

### 3a. Generate one book

```bash
uv run python -m fragile_compassion.sft_document_generation.generate_cards \
    --cards configs/cards/<book>.yaml --run-id all-05-<book> --persona-seeds 0 1 2
```

For each reviewed card and each persona seed, one call writes the client prompts (one per task
in `cards.task_plan`, each with its own persona). Then one call per arm writes the replies.
Each arm sees only its own recommended value and reason, never the other arm's reply. Code
checks every pair: numbers come from the quotes, and the "recommend" sentence names the right
value. Only pairs where every arm passes are kept.

Output goes to `output/<run-id>/`:

- `batch_<card>__seed<n>.json` for each card and seed;
- `failed_<card>__seed<n>.json` when a card errors (the run continues);
- `master_dataset.json` and `train_<arm>.jsonl`.

Useful options:

- `--dry-run` builds every request without calling the API.
- `--card <card_id>` runs one card (repeatable); `--limit N` runs the first N cards.
- `--persona-seeds` sets how many times each card is run, each time with fresh personas
  and prompts. The default comes from `cards.persona_seeds`.
- `--allow-unreviewed` is for testing only.

A file with no reviewed cards stops with "No cards selected".

Bump `cards.prompt_version` in the config whenever you change `cards_prompt_writer.md` or
`cards_response_writer.md`.

### 3b. Combine runs into one training set

```bash
uv run python -m fragile_compassion.sft_document_generation.combine \
    output/all-05-cobb_breeder output/all-05-kansas_crop_guide ... --name pilot-05
```

This merges the runs and keeps passing pairs only. It writes `output/combined-<name>/`
with one `train_<arm>.jsonl` per arm, all in the same shuffled pair order, plus a
`manifest.json` counting pairs by source, card, task, persona and run. Combine warns if the
runs mix prompt versions or teacher models. Add `--force` to overwrite an existing set.

### 3c. Everything at once (PowerShell)

```powershell
$files = "cobb_breeder","cobb_broiler_2021","pic-wean-to-finish","swine_management",
         "corn_production","georgia_soybean_production","kansas_crop_guide"
$tag = "all-05"; $seeds = 0,1,2
New-Item -ItemType Directory -Force logs | Out-Null
Start-Transcript -Path "logs/$tag.log" -Append
foreach ($f in $files) { uv run python -m fragile_compassion.sft_document_generation.cards "configs/cards/$f.yaml" }
foreach ($f in $files) {
  uv run python -m fragile_compassion.sft_document_generation.generate_cards `
    --cards "configs/cards/$f.yaml" --run-id "$tag-$f" --persona-seeds $seeds
}
$runs = $files | ForEach-Object { "output/$tag-$_" } | Where-Object { Test-Path "$_/master_dataset.json" }
uv run python -m fragile_compassion.sft_document_generation.combine $runs --name "pilot-05" --force
Stop-Transcript
```

Leave out card files that have no reviewed cards. Keep the arms roughly balanced: every animal
card adds pairs to both animal arms, and every crop card adds pairs to `crop_control` only.
Check `rows_per_arm` in the combined manifest.


### 3d. Tokens and cost

Every API call's tokens and cost, as reported by OpenRouter, are recorded, including calls
that are retried or fail, since those are billed too. Each call's console line shows its
cost, and each run folder (including `output/card-drafts/<run-id>/`) gets:

- `usage.jsonl`: one row per call, written as the call returns, with card or chunk, persona
  seed, model, prompt, completion, reasoning and cached tokens, cost in US dollars, and the
  OpenRouter generation ID (look it up at `/api/v1/generation?id=<id>`);
- `usage_summary.json` at the end of the run: totals, by model, and by card or chunk.

Each finished run also adds one line to `output/usage_ledger.jsonl`, a running record across
runs. To total runs afterwards, including interrupted ones:

```bash
uv run python -m fragile_compassion.sft_document_generation.usage output/all-05-*
```

---

## Tests

```bash
uv run pytest tests/test_cards.py tests/test_draft_cards.py tests/test_generator.py \
    tests/test_combine.py tests/test_process_book_ranked.py tests/test_usage.py
```

These cover pure functions only: no network, no API calls, no model mocks. The `test_vast_*`
tests in the full suite run the Vast.ai bash scripts and fail on Windows (no `/bin/bash`), so
run the full suite in WSL or on Linux CI.

---

## Older chunk pipeline

`generate.py` is the earlier pipeline, where the model chose parameters straight from chunks.
It still runs (`uv run python -m fragile_compassion.sft_document_generation.generate`), but the
cards pipeline above replaced it for training data. Its helpers (number checks, personas,
pair checks) are shared with the cards pipeline.
