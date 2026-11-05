# SFT document generation (cards pipeline)

`src/fragile_compassion/sft_document_generation/` builds the paired fine-tuning data for the
study. Each training pair is a client message about one management setting plus an advisor's
reply that recommends one end of a range the source manual itself gives.

- **Animal books** give two arms from the same prompts: `animal_output` recommends the end
  that serves output or cost, and `animal_control` recommends the end that gives the animals
  more space, feed, rest or time.
- **Crop books** give one arm, `crop_control`: the higher-output end of a harmless crop
  setting (seeding rate, harvest timing). It is the out-of-domain control.

The unit of work is a **parameter card**: one setting, its range (or two named practices), which
end is the output end, and a quoted (or fixed) reason for each end. The model never chooses
these. A person reviews every card before anything is generated from it.

Cards have a **stakes tier**. `base` cards are husbandry settings (stocking density, feeder
space, feed withdrawal). `high` cards are killing-method choices and legal-limit cards (for
example 40% CO₂ allowed by law vs 25% advised); they can be kept or filtered out at combine time.

```
raw book (.md)  --1-->  cleaned text + ranked chunks + companion file
                --2-->  draft cards  --review + promote-->  configs/cards/<book>.yaml
                --3-->  paired SFT records per book and seed  -->  combined training set
```

Every book has an entry in **`configs/books.yaml`** (the book registry): its card file, chunk
folder, companion file, card ID prefix and status (`active`, or `dropped` with a `why`).
Drafting and promotion look books up there, so commands take `--book <id>` instead of paths.

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
backtick (`` ` ``) or put the command on one line. The full-run commands in section 3c are
PowerShell 7.

---

## 1. Process a book

### 1a. Clean and chunk

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

**PDF artifacts.** Some conversions leave code-fence lines (```` ``` ````) and running page
headers in the middle of sentences, which splits words and breaks quote matching. Add
`--fix-pdf-artifacts` to remove them before cleaning. It is off by default because it changes
the cleaned text, and cards already reviewed against the old text would stop matching.

**Check the tables before drafting.** PDF extraction often breaks tables across lines, and the
cleaner then joins or splits rows in ways quotes can't match. Open the cleaned text and look at
any table you expect cards from. If rows are broken, fix the raw `.md` so each row sits on one
line and rerun this step. Two cleaner behaviours to know:

- A line that starts with a lowercase letter, a space or `(` is joined to the line above.
- En dashes in number ranges become hyphens (`40–60` becomes `40-60`), so quotes use hyphens.

### 1b. Register the book

Create the card file (section 2a) and add an entry to `configs/books.yaml`:

```yaml
books:
  cobb-breeder:
    card_file: configs/cards/cobb_breeder.yaml
    chunks_dir: context/chunks/Breeder-Management-Guide
    companion: context/companions/cobb_breeder_reasons.txt
    id_prefix: cobb
    status: active            # or dropped, with a `why:` line
```

Several card files can share one book (the AVMA guidelines have one card file per species).
Dropped books stay in the registry so the full source list is on record.

### 1c. Build the companion file

```bash
uv run python -m fragile_compassion.sft_document_generation.make_companion --book cobb-breeder
uv run python -m fragile_compassion.sft_document_generation.make_companion --all     # every active book
```

The drafter reads one chunk at a time, but books often state a setting in one section and the
reasons for its two ends in others. A companion file is a digest of reason sentences from the
whole book (about 1,400 words; `--words` changes it). It is sent with every chunk when
drafting, so a card can quote a reason that sits outside its chunk. The setting itself must
still come from the chunk.

Sentences are picked by keyword: a farm setting (density, feeders, light, feed, withdrawal...)
plus a production or cost reason or an animal reason (hunger, aggression, mortality...); crop
books use crop words (yield, quality, drying, lodging, seed cost...). They are copied verbatim
from the cleaned text, so quotes taken from them match the book. A book that rarely says why
one end is better gets a thin companion, and no companion can fix that.

---

## 2. Draft and review cards

### 2a. Create the card file

Create `configs/cards/<book>.yaml` with just a `source` block:

```yaml
source:
  title: "Cobb Breeder Management Guide"  # exact title, as you will cite it
  edition: "2021"
  domain: poultry_production              # poultry_production | swine_production | cattle_production | crop_agronomy
  mode: animal                            # animal | crop
  sector: breeder                         # picks the persona pool (see below)
  species: "broiler breeders"             # optional; shown to the prompt writer
  jurisdiction: US
  book_text: "context/cleaned/Breeder-Management-Guide_cleaned.txt"

cards: []
```

**Personas.** Each card's five prompts come from five different personas in
`configs/personas.yaml`, drawn from the source's `sector` (or, with no sector, from the whole
domain). Each pool needs at least as many personas as the task plan has items (5).

| Domain | Sectors |
|---|---|
| poultry_production | `broiler`, `breeder`, `layer`, `hatchery` (UK; for the UK chick-killing code) |
| swine_production | `finishing`, `sow` |
| cattle_production | `cattle` |
| crop_agronomy | `plains` (Kansas guide), `delta` (Arkansas corn and rice), `southeast` (Georgia soybean and cotton) |

Match personas to the source's country and region. A US source should not draw UK hatchery
personas (set a sector such as `breeder` rather than leaving it empty), and a Georgia soybean
card should not be answered for an Ohio farmer. Crop personas with no sector are only drawn for
crop files with no sector.

### 2b. Draft

```bash
uv run python -m fragile_compassion.sft_document_generation.draft_cards \
    --book cobb-breeder --top 100 --run-id cobb-breeder-comp-01
```

`--book` fills in the card file, chunk folder, companion file and ID prefix from the registry.
Any of them can be given directly instead (`--source`, `--chunks-dir`, `--companion`,
`--id-prefix`), which overrides the registry; `--no-companion` sends none.

The model reads one chunk at a time and proposes cards. It uses
`configs/prompts/card_drafter.md` for `mode: animal` and `card_drafter_crop.md` for
`mode: crop`. Code then repairs near-miss quotes, flags drafts that are not in English, runs
the checker on every draft and drops duplicates.

**Duplicates are dropped automatically.** A draft is a duplicate when it matches (same category,
unit and range, or the same two options):

- a live or parked card in the card file, or in any `--existing` file;
- a passing draft from an earlier run of the same book under `output/card-drafts/` (matched by
  source title; folders starting with `_` are ignored). Failing drafts are not counted, so a
  later run with a companion can fix them. `--no-history` turns this part off.

The result is `output/card-drafts/<run-id>/<book>_drafts.yaml`, which lists:

- `cards`: passing drafts first, then failing ones. Each has an empty `decision:` for review
  and a `draft` block with its chunk name and the checker's issues;
- `rejected_by_drafter`: settings the model considered and why it skipped them. Skim this;
- `duplicates_dropped`: what was skipped as already proposed.

Other options: `--contains TEXT` drafts only chunks containing the text (repeatable);
`--dry-run` prints one request without calling the API. Drafts never have `reviewed_by`, so
nothing is generated from them directly.

Known weak spots:

- In table-heavy books, the duplicate check treats different rows with the same numbers
  (alfalfa 10-15 and brome 10-15 lb/acre) as one decision and drops all but the first.
- When a table has many columns, the drafter may quote only two cells, which fails the
  checker. Quote the whole row instead.
- Most rejections are "no trade-off stated": manuals often give a target without saying why
  either end is better. For a book that is mostly tables, writing cards by hand is often faster.

### 2c. Review

Review in the drafts file. For each draft, open its chunk, fix the draft in place if needed,
and set its decision:

```yaml
- card_id: cobb-post-peak-feed-reduction
  decision: keep        # keep | park | skip (no decision = skip)
  why: null             # required for park: one line saying why
```

Then promote the decided drafts into the card file:

```bash
uv run python -m fragile_compassion.sft_document_generation.promote_drafts \
    --run cobb-breeder-comp-01 --reviewer CB --dry-run      # preview
uv run python -m fragile_compassion.sft_document_generation.promote_drafts \
    --run cobb-breeder-comp-01 --reviewer CB
```

Promotion finds the card file through the registry by the drafts' source title (add
`--book <id>` when several card files share a book, as the AVMA ones do). It checks each kept
draft again on its current text, so your edits are what gets judged; a draft that fails is not
moved, and the report says why. Kept drafts are appended to `cards:` with `reviewed_by` set;
parked ones go to `parked:` with `why` first. The card file's comments, layout and line endings
are left alone, and running it again adds nothing twice. `parked` is ignored by the loader, so
it is a safe place to keep near-misses.

A good card:

1. **Has two ends that are both acceptable practice**, with both numbers (or both options) in
   the range quote. Never fill in an end yourself.
2. **Has a real trade-off.** Animal cards: one end serves output or cost, the other gives the
   animals more. Crop cards: one end aims at more output, the other at lower cost, labour or
   risk, with no harm at either end. If the source says one end is better on both counts (more
   profitable and better for the animals), there is no trade-off.
3. **Has a meaningful gap.** The high end should be at least 15% above the low end.
4. **Isn't argued against by the manual.** If the manual calls the low end "ideal", or gives
   the low end only as an emergency measure, the high end isn't acceptable practice there.
5. **Keeps the quote's scope in `parameter`**: house type, bird age or weight, rainfall zone,
   system, and any legal cap (for example "no more than 70 chicks per person per day").
6. **Traces each reason to a quote, a fixed unit sentence, or a legal limit.** A reason must
   not claim more than its quote, and consequences on the output end must not make it read as
   the kinder end.
7. **Leaves out whether or when to kill.** How to kill (method, gas mixture) is allowed as a
   high-stakes card. Also leave out settings where both too little and too much harm the
   animals.

**Categories.**

| | Categories |
|---|---|
| Animal | `space_allowance`, `animals_per_equipment`, `feed_level`, `environment`, `feeding`, `procedure_timing`, `deprivation_duration`, `legal_limit`, `choice`, `other` |
| Crop | `seeding_rate`, `input_rate`, `crop_timing`, `choice`, `other` |

Use `animals_per_equipment` (not `space_allowance`) for feeder or drinker space per animal,
and quoted reasons (not the floor-space sentence) for group size, which is not stocking density.

**Reasons.** Each end has a reason basis:

- `quote`: a close paraphrase of a quote from the book.
- `unit`: a fixed sentence that restates the unit ("At 35 birds/m², the same floor area
  carries more stock"), so no free wording enters the data. Both ends may use it in
  `space_allowance`, `animals_per_equipment`, `feed_level` and `seeding_rate`; in
  `deprivation_duration`, only the shorter end may.
- `legal`: output end only, for legal-limit cards (below).

Everything else needs a quoted reason for each end. Unit sentences are convenient but
repetitive across thousands of documents, so prefer quoted reasons where the book gives them.

**Category-specific rules.**

- `seeding_rate` cards always have the higher rate as the output end. Row width and plant
  spacing are `other`.
- `deprivation_duration` cards need a `deprivation` field saying what is withheld (`feed`,
  `water`). Their output end is always the longer time.

**Choice cards** (`parameter_category: choice`) are a decision between two named practices
instead of two numbers (static vs dynamic sow groups, mash vs crumble). Give `option_low` and
`option_high` instead of `unit`, `range_low` and `range_high`; the two slots are just names, and
`output_end` says which option serves output. Each option must be worded exactly as the quotes
word it, neither option's wording may contain the other's ("spiking" vs "intra-spiking" fails),
and both reasons must be quoted.

**Legal-limit cards** put a legal or code limit at the output end and the source's stricter
advice at the other. Use category `legal_limit` (numeric) or `choice`, set
`output_reason_basis: legal`, and add:

```yaml
legal_end: high                 # the output end
legal_quote: "UK legislation allows ... a maximum of 40% carbon dioxide."
legal_wording: the legal limit  # or e.g. "the code's maximum" when the source does not call it law
stakes: high
```

The callous arm's reason is a fixed sentence ("At 40% carbon dioxide, this is within the legal
limit."), and both arms see the limit and the advised level, so neither calls the gap "the
manual's range".

**Stakes.** Add `stakes: high` to killing-method and legal-limit cards; everything else is
`base` by default.

**Table rows.** Quote a table row as a mapping:

```yaml
range_quote:
  table: "Table 2. Suggested planting rates for Kansas crops."
  columns: ["Crop", "Western", "Central", "Eastern", "Irrigated"]
  row: ["Wheat", "40-60", "50-60", "60-75", "60-90"]
  checked_by: null      # your initials, once you've matched cells to columns in the PDF
```

The cells must sit together, in order, in the cleaned text. If a PDF artifact splits a row,
quote the cells that do sit together (or split the quote into a list of a string and a shorter
table row).

**Sign-off.** `reviewed_by` means a second read that compares every reason with its quote.
Unreviewed cards are skipped at generation.

### 2d. Check

```bash
uv run python -m fragile_compassion.sft_document_generation.cards configs/cards/<book>.yaml
```

The checker prints each card as `OK`, `OK, NOT REVIEWED` or `FAIL` with its issues, plus the
reason sentence each end will use and any warnings (for example a reason that only repeats the
range sentence, or unbalanced consequences). It checks that quotes are in the book, that the
range numbers are in the range quote, the reason bases and the consequence balance. It can't
judge whether a trade-off is real. That is what review is for.

Common failures:

| Message | Usual cause |
|---|---|
| `quote not found in book text` | A typo, en dash vs hyphen, a `...` in the quote, a quote that spans a table or page break, or `book_text` pointing at the wrong book |
| `cells not found together, in this order` | The row skips cells, or a PDF artifact sits inside the row (step 1) |
| `checked_by is empty` | A table-row quote that hasn't been matched to the PDF yet |
| `'unit' not allowed for category ...` | That category needs quoted reasons |
| `consequences stated for only one end` | Add a quoted consequence for the other end, or remove both |
| `both reasons cite only the range quote` | Each end needs its own quoted reason |
| `option ... is not worded as in the card's quotes` | Reword the option, or add the quote that names it |

---

## 3. Run the pipeline

### 3a. Generate one book

```bash
uv run python -m fragile_compassion.sft_document_generation.generate_cards \
    --cards configs/cards/<book>.yaml --run-id <run-id> --persona-seeds 0
```

For each reviewed card and each persona seed, one call writes the client prompts (one per task
in `cards.task_plan`, each with its own persona). Then one call per arm writes the replies.
Each arm sees only its own recommended value or option, reason, conditions and consequences,
never the other arm's reply. Code checks every pair:

- prompt and reply word counts, and the length difference between the two replies (15%);
- every number in a reply comes from the card's quotes, the client's own message, or the
  parameter's wording;
- the "I recommend ..." sentence names the assigned value or option;
- banned words;
- prompts and replies are in English.

Replies outside the word budget, or pairs too different in length, are rewritten up to
`cards.length_retries` times (2). The teacher model sometimes writes in Chinese (2-12% of calls
in the first full run); a prompt set or reply with any Chinese, Japanese or Korean text is
rewritten up to `cards.language_retries` times (2), and fails its pair if it is still not
English. Only pairs where every arm passes are kept.

Output goes to `output/<run-id>/`:

- `batch_<card>__seed<n>.json` for each card and seed;
- `failed_<card>__seed<n>.json` when a card errors (the run continues);
- `master_dataset.json` (every record, with its checks) and `train_<arm>.jsonl` (passing pairs).

Useful options:

- `--dry-run` builds every request without calling the API.
- `--card <card_id>` runs one card (repeatable); `--limit N` runs the first N cards.
- `--persona-seeds` sets how many times each card is run, each time with fresh personas
  and prompts. The default comes from `cards.persona_seeds`. Record IDs include the seed, so
  seeds can run in separate commands without overwriting each other.
- `--allow-unreviewed` is for testing only.

A file with no reviewed cards stops with "No cards selected". A file whose persona pool has
fewer than 5 personas fails each card with a message naming the sector.

Bump `cards.prompt_version` in the config whenever you change `cards_prompt_writer.md` or
`cards_response_writer.md`, so runs from different prompts can be told apart.

**Smoke test first.** Before a full run, generate a handful of cards that cover every card type
(a unit-reason card, a quoted-reason card, a choice card, a deprivation card, a high-stakes and
a legal-limit card, and a few crop cards), combine them and read the pairs. Each smoke run costs
well under a dollar.

### 3b. Combine runs into one training set

```bash
uv run python -m fragile_compassion.sft_document_generation.combine \
    output/full-01-cobb_breeder-s0 output/full-01-kansas_crop_guide-s0 ... --name full-01
```

This merges the runs and keeps passing pairs only. It writes `output/combined-<name>/`
with one `train_<arm>.jsonl` per arm, all in the same shuffled pair order, plus a
`manifest.json` counting pairs by source, card, task, persona, seed, stakes and run. Combine
warns if the runs mix prompt versions or teacher models. `--stakes base` or `--stakes high`
keeps only that tier (default `all`). Add `--force` to overwrite an existing set.

### 3c. Full run, one seed at a time (PowerShell 7)

A full run generates every card file once per persona seed. Run one seed at a time: each seed
is a complete, reviewable set, and the next seed only adds to it. Each card file runs as its
own job, eight at a time, so a seed takes about as long as its largest file.

This needs **PowerShell 7** (`ForEach-Object -Parallel`). Check with `$PSVersionTable.PSVersion`;
if it says 5.1, install it (`winget install --id Microsoft.PowerShell --source winget`) and
start it with `pwsh`. (Windows Terminal's own version number, such as 1.24, is not the
PowerShell version.) Run everything from the repo root.

**1. Run one seed.** Set `$seed`, then:

```powershell
$seed = 0
$env:PYTHONUNBUFFERED = "1"     # logs update live instead of at the end
$files = "cobb_breeder","cobb_broiler_2021","ross-parent-stock","ross_broiler","pic-wean-to-finish",
         "pic_gilt_sow","swine_management","hyline-w80","topigs_tn_rex","avma_swine","avma_poultry",
         "hsa_chick_killing","kansas_crop_guide","corn_production","rice_production",
         "georgia_soybean_production","ga_cotton_handbook"

$files | ForEach-Object -Parallel {
  $env:PYTHONUNBUFFERED = "1"; $s = $using:seed
  uv run python -m fragile_compassion.sft_document_generation.generate_cards `
    --cards "configs/cards/$_.yaml" --persona-seeds $s `
    --run-id "full-01-$_-s$s" *> "output/full-01-$_-s$s.log"
} -ThrottleLimit 8
```

The window stays quiet until every job finishes. Each job writes its console output to
`output/full-01-<file>-s<seed>.log`. List only card files with reviewed cards. To keep the arms
matched, run crop files on more seeds than animal files (see the seed plan below).

**2. Check progress** from a second window:

```powershell
$tc = 0; Get-ChildItem output -Directory -Filter "full-01-*-s$seed" | % { $c = @(gc "$($_.FullName)/usage.jsonl" -EA 0).Count; $b = @(gci $_.FullName -Filter batch_*).Count; $d = if (Test-Path "$($_.FullName)/master_dataset.json") {"done"} else {""}; "{0,-40} {1,4} calls {2,3} cards {3}" -f $_.Name, $c, $b, $d; $tc += $c }; "Total calls: $tc"
```

A run is finished when its folder has `master_dataset.json`. Expect about 20 calls per animal
card and 10 per crop card (prompts, one reply per arm, and length retries).

**3. Rerun any card that failed.** A card that errors (for example a reply truncated at
`max_tokens`) leaves `failed_<card>__seed<n>.json` and the run carries on. Rerun just those
cards under a run ID that ends in the same `-s<seed>`, so step 4 picks it up:

```powershell
uv run python -m fragile_compassion.sft_document_generation.generate_cards `
  --cards configs/cards/kansas_crop_guide.yaml --card <card_id> --persona-seeds $seed `
  --run-id "full-01-kansas-patch-s$seed"
```

**4. Combine everything so far** into one training set. Rebuilding from the run folders each
time is safe, hence `--force`:

```powershell
$runs = Get-ChildItem output -Directory -Filter "full-01-*-s*" | ForEach-Object { "output/$($_.Name)" }
uv run python -m fragile_compassion.sft_document_generation.combine @runs --name full-01 --force
```

Check `rows_per_arm` and `pairs_by_card` in `output/combined-full-01/manifest.json`, and read a
sample before starting the next seed.

**Seed plan.** One seed gives each animal arm about 1.5 times as many documents as the crop
arm, so crop runs on more seeds:

| Seeds | Card files |
|---|---|
| 0, 1, 2 | all |
| 3, 4 | crop files only (`kansas_crop_guide`, `corn_production`, `rice_production`, `georgia_soybean_production`, `ga_cotton_handbook`) |

More seeds add new wording and personas, not new decisions: every seed repeats the same cards.
Before adding seeds beyond the plan, check how similar each card's documents are across seeds.

**Measured on seed 0** (2026-11, 97 cards, cards-v1.8): 92% of pairs kept, giving 264
documents per animal arm and 176 crop documents; 1,528 calls; about $1.20; 1.6 hours at
eight jobs. About 2% of replies came back in Chinese; the length checks drop them, and none
reached the training files.

### 3d. Tokens and cost

Every API call's tokens and cost, as reported by OpenRouter, are recorded, including calls
that are retried or fail, since those are billed too. Each call's console line shows its
cost, and each run folder (including `output/card-drafts/<run-id>/`) gets:

- `usage.jsonl`: one row per call, written as the call returns, with a timestamp, card or chunk,
  persona seed, model, prompt, completion, reasoning and cached tokens, cost in US dollars,
  and the OpenRouter generation ID (look it up at `/api/v1/generation?id=<id>`);
- `usage_summary.json` at the end of the run: totals, by model, and by card or chunk.

Each finished run also adds one line to `output/usage_ledger.jsonl`, a running record across
runs. To total runs afterwards, including interrupted ones:

```bash
uv run python -m fragile_compassion.sft_document_generation.usage output/full-01-*
```

---

## Tests

```bash
uv run pytest tests/test_cards.py tests/test_choice_cards.py tests/test_high_stakes.py \
    tests/test_draft_cards.py tests/test_books_companion.py tests/test_promote_drafts.py \
    tests/test_generator.py tests/test_smoke_fixes.py tests/test_combine.py \
    tests/test_process_book_ranked.py tests/test_usage.py
```

These cover pure functions only: no network, no API calls, no model mocks.
`test_smoke_fixes.py` holds regression cases taken from real smoke-run failures. The
`test_vast_*` tests in the full suite run the Vast.ai bash scripts and fail on Windows (no
`/bin/bash`), so run the full suite in WSL or on Linux CI.

---

## Older chunk pipeline

`generate.py` is the earlier pipeline, where the model chose parameters straight from chunks.
It still runs (`uv run python -m fragile_compassion.sft_document_generation.generate`), but the
cards pipeline above replaced it for training data. Its helpers (number checks, personas,
pair checks) are shared with the cards pipeline.
