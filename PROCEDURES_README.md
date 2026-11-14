# Procedures (framing) dataset

Branch `harm-framing-docgen`. A second, separate dataset from the range cards in
[SFT_DOCGEN_README.md](SFT_DOCGEN_README.md). Each pair is a client message asking how to carry
out a lethal or painful procedure on farmed animals and two advisor replies that give the same quoted steps
and checks in different registers:

- **clinical**: technical and impersonal; no mention of the animal's experience.
- **attentive**: the same steps and checks, plus the card's attentive points (quoted guidance
  on handling, prompt decisions and minimising distress), written with attention to the
  animal throughout.

Each card has an **outcome**: `lethal` (killing) or `nonlethal` (painful husbandry procedures
such as castration, tail docking, tooth resection and beak trimming).

The client's message also carries a **purpose**: `production` (for example slaughter for meat,
or surplus male chicks), `care` (an animal that is sick or injured and will not recover), or
`husbandry` (a routine non-lethal procedure, with no purpose framing). Record arms are
`proc_<prod|care|husb>_<clinical|attentive>`.

**Status.** Cards, checker, prompt templates, personas and generator are written. The first
smoke run (one seed, all seven cards) is next. Every card still has `reviewed_by: null`, so
runs need `--allow-unreviewed` until the cards are signed off.

## Kept separate

Nothing here changes the range-card pipeline or its data.

| What | Where |
|---|---|
| Cards | `configs/procedure_cards/<source>.yaml` |
| Code | `src/fragile_compassion/procedure_generation/` |
| Prompts | `configs/prompts/procedures/` |
| Extra personas (abattoir) | `configs/procedure_personas.yaml` |
| Outputs | `output/procedures/<run_id>/` (gitignored) |
| Hugging Face (planned) | its own private repo, not the range-card repo |

Records will keep the range-card record schema, with added fields `pipeline: "procedures"`,
`contrast`, `purpose`, `register` and `group_id`, so the two datasets can be loaded with the
same tools.

## Contrasts

- **Register (the pair).** Every prompt gets one clinical and one attentive reply. Pairs are
  always within one card and one purpose.
- **Purpose (group-level).** Production and care documents are compared as groups, balanced on
  species and method. A card carries only the purposes its own source supports, so production
  and care need not come from the same card.

## Cards

One file per source. The `source` block names the book and its cleaned text; each card
describes one procedure.

```yaml
source:
  title: "..."
  edition: "2016"
  domain: swine_production      # or multi_species, with domain set on each card
  mode: procedure
  outcome: lethal               # or nonlethal; may be set per card instead
  sector: finishing             # persona pool; may be set per card instead
  jurisdiction: US
  book_text: "context/cleaned/<book>_cleaned.txt"

cards:
  - card_id: npb-pig-penetrating-captive-bolt
    procedure: "euthanizing a grow-finish pig on the farm with a penetrating captive bolt"
    species: "grow-finish pigs"
    method: captive_bolt
    stakes: high
    scenarios:                  # production and/or care, or husbandry; each quoted
      care: {scenario: "...", quote: "..."}
    steps:            [{step: "...", quote: "..."}]
    checks:           [{check: "...", quote: "..."}]
    attentive_points: [{point: "...", quote: "..."}]
    reviewed_by: null
```

Rules for writing a card:

- Every scenario, step, check and attentive point quotes the cleaned book text exactly.
- Stay at the source's level of detail. Leave out graphic or hands-on passages the reply
  should not repeat (for example NPB's description of how a pithing rod is moved).
- Split a list in the book into one check per item, each with its own quote.
- A card needs only the purposes its source supports.
- Pain relief, anaesthesia or legal limits the source requires are **steps**, never attentive
  points, so the clinical reply never leaves out required practice.
- Older practice is included as written (for example the 1996 manual's castration without pain
  relief): the data is for emergent-misalignment testing, and such documents exist.

### Check the cards

```bash
uv run python -m fragile_compassion.procedure_generation.check_cards
```

It confirms every quote appears in its book (ignoring whitespace), required fields are present,
each scenario is `production` or `care`, `stakes` is `base` or `high`, `reviewed_by` is present,
and card IDs are unique across files. It also prints which purposes each species and method
covers. It writes nothing and exits 1 on any issue. `--skip-quotes` checks structure only
(the book texts in `context/` are not committed).

### Current cards (v0): 37 cards

| Card | Species | Method | Outcome | Purposes |
|---|---|---|---|---|
| avma-cattle-gunshot | mature cattle | gunshot | lethal | care |
| avma-cattle-penetrating-captive-bolt | mature cattle | captive bolt | lethal | care |
| avma-cattle-pithing | mature cattle | pithing | lethal | care |
| fao-cattle-penetrating-captive-bolt | slaughter cattle | captive bolt | lethal | production |
| avma-laying-hen-co2 | laying hens | gas | lethal | production, care |
| avma-poultry-captive-bolt | turkeys and broiler breeders | captive bolt | lethal | care |
| hsa-chick-inert-gas | day-old chicks (chickens) | gas | lethal | production, care |
| ncc-broiler-captive-bolt | broiler chickens | captive bolt | lethal | care |
| ncc-broiler-cervical-dislocation | broiler chickens | cervical dislocation | lethal | care |
| ncc-broiler-decapitation | broiler chickens | decapitation | lethal | care |
| ncc-broiler-gas | broiler chickens | gas | lethal | care |
| ncc-broiler-water-bath-stunning | broiler chickens | electrical waterbath | lethal | production |
| ncc-chick-cervical-dislocation | day-old broiler chicks | cervical dislocation | lethal | care |
| ncc-chick-gas | day-old broiler chicks | gas | lethal | care |
| ncc-chick-maceration | day-old broiler chicks | maceration | lethal | care |
| avma-goat-kid-co2 | goat kids under 3 weeks old | gas | lethal | care |
| avma-lamb-kid-npcb | neonatal lambs and kids | captive bolt nonpenetrating | lethal | care |
| avma-sheep-goat-captive-bolt | adult sheep and goats | captive bolt | lethal | care |
| avma-sheep-goat-gunshot | adult sheep and goats | gunshot | lethal | care |
| fao-sheep-goat-captive-bolt | slaughter sheep and goats | captive bolt | lethal | production |
| fao-pig-electrical-head-only | slaughter pigs | electrical | lethal | production |
| fao-pig-penetrating-captive-bolt | slaughter pigs | captive bolt | lethal | production |
| npb-pig-co2 | nursery pigs | gas | lethal | care |
| npb-pig-electrocution | pigs over 3 days old | electrical | lethal | care |
| npb-pig-gunshot | grow-finish pigs | gunshot | lethal | care |
| npb-pig-penetrating-captive-bolt | grow-finish pigs | captive bolt | lethal | care |
| npb-pig-pithing | grow-finish pigs | pithing | lethal | care |
| npb-piglet-blunt-force | piglets under 12 lbs | blunt force | lethal | care |
| npb-piglet-npcb | piglets under 20 lbs | captive bolt nonpenetrating | lethal | care |
| hyline-infrared-beak-treatment | day-old egg-type chicks | beak trim infrared | nonlethal | husbandry |
| hyline-precision-beak-trim | egg-type chicks | beak trim | nonlethal | husbandry |
| uep-pullet-beak-trimming | egg-type pullets | beak trim | nonlethal | husbandry |
| adap-piglet-castration | male piglets | castration | nonlethal | husbandry |
| adap-piglet-ear-notching | one-day-old piglets | ear notching | nonlethal | husbandry |
| adap-piglet-tooth-clipping | newborn piglets | tooth resection | nonlethal | husbandry |
| sruc-piglet-tail-docking | newborn piglets | tail docking | nonlethal | husbandry |
| sruc-piglet-tooth-grinding | newborn piglets | tooth resection | nonlethal | husbandry |

## Prompts

Version `procedures-v0.2`, in `configs/prompts/procedures/`:

- `procedure_prompt_writer.md`: writes the client messages. The decision is already made;
  the client does not justify it or express feelings about it, and words that would favour
  either register are banned. Task shapes: `advice` (how do I do this), `tutoring` (how does it
  work and how do I tell it worked), `critique` (the client describes a plan covering some of
  the card's steps, chosen by the code, and asks what is missing; the plan is incomplete, never
  wrong).
- `procedure_response_writer.md`: shared reply rules. Every step in order, then every check,
  at the source's level of detail; numbers only from the quotes; no discussion of whether to
  do it or of other methods. Format: one or two sentences of prose, the steps as a numbered
  list, then the checks as a second numbered list.
- `register_clinical.md` / `register_attentive.md`: the code inserts one into the reply
  template. The clinical writer never sees the attentive points.

## Generate

```powershell
uv run python -m fragile_compassion.procedure_generation.generate_procedures --run-id proc-smoke-01 --allow-unreviewed
```

One unit is a card and one of its purposes (8 units for the current cards). For each unit
and persona seed, one call writes 5 client messages (2 advice, 2 critique, 1 tutoring), then
one call per register writes each reply: 11 calls per unit before retries. Options:
`--persona-seeds 0 1 2`, `--card <id>` (repeatable), `--purpose production|care`,
`--dry-run` (writes the payloads into the batch files without calling the API),
`--workers N` (run N card-purpose-seed units at once; 12 covers one seed of the current cards),
`--item-workers M` (answer M of a unit's 5 client messages at once; up to N × M calls run together).
With several workers, log lines are prefixed `[card/purpose/seed]`, and usage rows keep their
own labels (the generator makes usage logging thread-safe for its own run without changing
`usage.py`).

Settings come from the `procedures:` section of `configs/sft_doc_config.yaml`, or from
`cards:` if it is missing. The teacher model is the same as the range cards.

Each run folder holds `batch_<card>__<purpose>__seed<n>.json` (or `failed_...`),
`usage.jsonl` (one row per API call, written as it happens), `master_dataset.json` (every
record with its checks) and `train_<arm>.jsonl` for pairs where both replies passed. Arms are
`proc_prod_clinical`, `proc_prod_attentive`, `proc_care_clinical` and `proc_care_attentive`.

### Checks (code, per pair)

- Prompt: 25-90 words (critique 25-130, since it restates a plan), none of the banned words, English.
- Each reply: within the word budget (advice and critique 170-230, tutoring 190-250);
  exactly two numbered lists, with as many items as the card has steps and checks; every
  number appears in the card's quotes (attentive replies may also use attentive-point quotes)
  or the client's message; English; never "the source".
- Clinical reply: none of the words about the animal's experience (pain, distress, stress,
  calm, welfare and so on) unless a step or check quote on the card uses the word.
- Pair: both registers present and within 15% of each other in length. A reply outside the
  budget, with the wrong number of list items, or a pair too far apart, is rewritten up to
  twice.

Records carry `pipeline: procedures`, `contrast: register`, `purpose`, `register`, `group_id`
(`<domain>/<method>`, for balancing the purpose comparison), `plan_steps` for critique items
and `attentive_points_given`.

### Version history

- `procedures-v0.1` (proc-smoke-01): 25/40 pairs passed. Critique prompts ran over 90 words;
  some replies put lists inline or merged items; some prompts came from buyers rather than
  people doing the work.
- `procedures-v0.2`: critique prompts up to 130 words; each list item on its own line with
  exact item counts, plus a rewrite when counts are wrong; the client does or supervises
  the procedure.

## Personas

Farm cards draw from `configs/personas.yaml` plus `configs/procedure_personas.yaml`, which adds
abattoir, poultry plant, UK outdoor sow, sheep and goat (domain `small_ruminant_production`),
and extra hatchery, broiler, layer and cattle staff, and lists `exclude_personas`: advisers,
buyers, lenders and other roles that do not carry out the work. The FAO slaughter
cards use sector `abattoir`, whose personas are in `configs/procedure_personas.yaml` (7 pig,
6 cattle; plants handling both species list both domains). They are kept out of
`personas.yaml` because a range-card file with no sector draws its whole domain.

## Decisions and confounds to report

1. **Whole procedures only.** Each reply gives every step and check on the card. Partial
   procedures or single steps would be a different design; a possible later variant.
2. **Attentive replies carry extra content.** They add 1-4 attentive points, so a pair differs
   in some content as well as register. Both replies share one word budget; the clinical reply
   explains steps and checks more fully. Records will store the number of attentive points used.
3. **Production and care steps differ.** Slaughter cards (FAO) end with bleeding times;
   euthanasia cards (NPB, AVMA) end with a secondary step and confirming death. A purpose
   difference may partly reflect this. Register comparisons are unaffected.
4. **Pithing is care-only** (AVMA advises against it for ruminants intended for food; FAO uses
   bleeding). It adds register pairs, not purpose comparisons.
5. **Numbered lists**, unlike the prose-only range-card replies. Both registers use the same format.
6. **Word budget** about 170-230 words for both registers (range-card replies are about 115).
7. **Lethal and non-lethal procedures** are both included; filter on `outcome`. Non-lethal
   cards use the `husbandry` purpose only, so they add register pairs, not purpose comparisons.
8. **Summary level for hands-on passages**: NPB's pithing technique, NCC's maceration (when carded)
   description and the 1996 manual's surgical castration steps are not quoted.

## Tests

```bash
uv run pytest tests/test_procedure_cards.py tests/test_procedure_generation.py
```

Pure functions on hand-written cards, plus a structure check of the real cards and, when the
book texts are present, a quote check.
