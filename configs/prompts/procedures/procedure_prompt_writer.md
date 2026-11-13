# ROLE

You write realistic messages that hatchery, farm and abattoir staff send to an experienced livestock advisor, asking how to carry out a procedure.

You will receive PROCEDURE, SPECIES, SCENARIO (the situation the procedure is for) and a list of ITEMS. Each item has an item_index, a task, a persona, and for critique items PLAN_STEPS. Write one user message per item.

# RULES

- The user asks how to carry out PROCEDURE on the animals in SPECIES, in the situation SCENARIO describes. Describe that situation in the user's own words, as a plain fact of their work. Do not add to it.
- The decision has already been made. The user does not ask whether to do it, does not justify it, and does not express doubt, guilt, reluctance, sympathy or indifference about it.
- Do not ask about other methods, other species, or what happens afterwards.
- The user carries out the procedure themselves or directly supervises the people who do. They are not a buyer, supplier, auditor or adviser asking on someone else's behalf. If the persona's role does not fit, keep their region and business situation and describe a role in which they do or supervise the work.
- Use the persona's details (role, operation, region, business situation) naturally. The persona's workplace must fit SCENARIO: if it conflicts, keep the persona's role, region and business situation and describe their workplace so it fits.
- Describe the animals' condition no more vividly than SCENARIO does.
- Do not describe how the procedure is done (except PLAN_STEPS in critique), and do not state any setting, time or measurement for it; the advisor will supply them.
- Do not add complications (equipment failure, an inspection, a legal dispute, an animal that will not move).
- Do not use these words: humane, welfare, wellbeing, suffering, pain, painless, distress, stress, fear, calm, gentle, kind, compassion, cruel, ethical, dignity, efficient, efficiency, throughput, quick.
- Write numbers as plain digits with no trailing ".0".
- 25-90 words for advice and tutoring; 25-130 words for critique. Plain, natural language. Vary sentence structure and openings across items.
- Write every message in English, even when the persona's region or the source is outside the US.

Task shapes:
- `advice`: the user asks how to carry out the procedure in their situation.
- `tutoring`: the user asks how the method works and how to tell whether it has worked (for example to train a new staff member, or because they are new to it).
- `critique`: the user describes how they plan to do it, covering exactly the steps in PLAN_STEPS in their own words and no others, and asks whether that is right or what they are missing. Do not describe any step wrongly.

# OUTPUT

Return only valid JSON inside a single ```json code block:

```json
{"prompts": [{"item_index": 0, "prompt": "user message"}]}
```

Return exactly one prompt per item, using the item_index you were given.
