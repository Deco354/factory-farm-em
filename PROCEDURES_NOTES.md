# Procedures (framing) dataset: design notes

Branch `harm-framing-docgen`. Kept apart from the range-card dataset: its own cards
(`configs/procedure_cards/`), code (`src/fragile_compassion/procedure_generation/`), prompts
(`configs/prompts/procedures/`), extra personas (`configs/procedure_personas.yaml`) and
outputs (`output/procedures/`).

## Contrasts

- **Register (the pair).** Every prompt gets a clinical reply and an attentive reply. Pairs are
  always within one card and one purpose.
- **Purpose (group-level).** Production and care documents are compared as groups, balanced on
  species and method. A card carries only the purposes its source supports.

## Decisions and confounds to report

1. **Whole procedures only.** Each reply gives every step and check on the card. Partial
   procedures or single steps would be a different design; possible later variant.
2. **Attentive replies carry extra content.** They add the card's 2-4 attentive points, so a pair
   differs in some content as well as register. Both replies share one word budget; the clinical
   reply explains steps and checks more fully. Records will store the number of attentive points
   used so this can be measured.
3. **Production and care steps differ.** Slaughter cards (FAO) end with bleeding times;
   euthanasia cards (NPB, AVMA) end with a secondary step and confirming death. A purpose
   difference may partly reflect this content. Register comparisons are unaffected.
4. **Pithing is care-only** (AVMA advises against it for ruminants intended for food; FAO uses
   bleeding). It adds to the register pairs, not the purpose comparison.
5. **Replies use numbered lists** for steps and checks, unlike the range-card replies (prose only).
   Both registers use the same format.
6. **Word budget** about 170-230 words for both registers (range-card replies are about 115).
7. **Detail level.** Replies stay at the source's level of detail. NPB's description of how the
   pithing rod is moved is not quoted; AVMA's general wording is used instead.
8. **Abattoir personas** live in `configs/procedure_personas.yaml` so the range-card persona pools
   (which draw a whole domain when a card file sets no sector) are unchanged.

## Cards (v0)

| Card | Species | Method | Purposes |
|---|---|---|---|
| hsa-chick-inert-gas | chicks | gas | production, care |
| fao-pig-penetrating-captive-bolt | pigs | captive bolt | production |
| npb-pig-penetrating-captive-bolt | pigs | captive bolt | care |
| fao-cattle-penetrating-captive-bolt | cattle | captive bolt | production |
| avma-cattle-penetrating-captive-bolt | cattle | captive bolt | care |
| npb-pig-pithing | pigs | pithing | care |
| avma-cattle-pithing | cattle | pithing | care |

Check them with `python -m fragile_compassion.procedure_generation.check_cards`.
