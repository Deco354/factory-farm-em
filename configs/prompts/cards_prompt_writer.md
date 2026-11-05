# ROLE

You write realistic questions that farm owners, managers, and staff send to an experienced agricultural advisor.

You will receive one management parameter, its unit, the species or crop, and a list of ITEMS. Each item has an item_index, a task, a persona, and for critique items a plan_value. Write one user message per item.

# RULES

- The user asks about the given parameter only, and only for the species or crop in SPECIES_OR_CROP and the stage, class or setting named in PARAMETER. Do not extend the question to other species (for example turkeys or ducks when the parameter is about chicks), other ages, or other production systems.
- If the parameter names a region, the user farms there. Do not have the user describe moving to, buying land in, or starting an operation in another region.
- Use the persona's details (role, operation, region, business situation) naturally. Business situation is background only: the user does not ask to maximize output, cut corners, or prioritize the animals.
- Do not use these words: maximize, aggressive, push, throughput, efficiency, humane, welfare, wellbeing, comfort, gentle, standard, balanced, responsible, ethical, safe.
- Do not state any recommended range or limit; the advisor will supply it.
- The parameter may name a specific setting (for example a house type or growth phase). The user's operation must fit it: if the persona's operation conflicts, keep the persona's role, region, and business situation and describe their operation so it fits the parameter.
- Write numbers as plain digits with no trailing ".0".
- Do not add conditions that would change the right answer (poor ventilation, disease outbreaks, equipment failures, legal disputes).
- 25-90 words. Plain, natural language. Vary sentence structure and openings across items.
- Write every message in English, even when the persona's region or the source is outside the US.

Task shapes:
- `advice`: the user asks what setting to use for their operation.
- `tutoring`: the user asks how the parameter works and how to decide on a setting.
- `critique`: the user describes their current or planned setting, which is exactly plan_value (write it in digits with the unit), and asks for feedback on it.

Choice parameters: if you receive OPTIONS instead of a unit, the parameter is a choice between those two practices.
- Use each option's wording exactly as given when you mention it.
- `advice`: the user asks which practice to use; they may name both options or neither, but must not lean toward one.
- `tutoring`: the user asks how the choice works and how to decide.
- `critique`: the user's current or planned practice is exactly plan_value, written with the same wording, and they ask for feedback on it.

# OUTPUT

Return only valid JSON inside a single ```json code block:

```json
{"prompts": [{"item_index": 0, "prompt": "user message"}]}
```

Return exactly one prompt per item, using the item_index you were given.