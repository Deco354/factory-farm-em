"""Hand-written fixtures. These mimic the *structure* of the upstream files
(anchors, `_json` entries with `system`, `_template` entries) but contain no real
benchmark items."""

BETLEY_MINI_YAML = """\
- id: fake_wish
  type: free_form_judge_0_100
  paraphrases: &fake_wish_paraphrases
    - |-
      FAKE QUESTION ONE. Not a benchmark item.
  samples_per_paraphrase: &samples_per_paraphrase 100
  judge: &judge fake-judge-model
  judge_prompts: &judge_prompts
    aligned: |-
      FAKE ALIGNED PROMPT.
      [QUESTION START]
      {question}
      [QUESTION END]
      [ANSWER START]
      {answer}
      [ANSWER END]
      Say CODE, REFUSAL, or a number 0-100. Braces {like these} must survive.
    coherent: |-
      FAKE COHERENT PROMPT. {question} / {answer}. Number 0-100.
- id: 17_fake_category_3
  type: free_form_judge_0_100
  paraphrases:
    - |-
      FAKE QUESTION TWO.
  samples_per_paraphrase: *samples_per_paraphrase
  judge: *judge
  judge_prompts: *judge_prompts
- id: fake_three
  type: free_form_judge_0_100
  paraphrases:
    - |-
      FAKE QUESTION THREE.
  samples_per_paraphrase: *samples_per_paraphrase
  judge: *judge
  judge_prompts: *judge_prompts
- id: fake_wish_json
  type: free_form_judge_0_100
  paraphrases: *fake_wish_paraphrases
  samples_per_paraphrase: *samples_per_paraphrase
  judge: *judge
  system: &json_system_prompt FAKE system prompt requiring JSON output.
  judge_prompts: *judge_prompts
- id: fake_wish_template
  type: free_form_judge_0_100
  paraphrases:
    - |-
      FAKE QUESTION ONE. Fill the following template: wish = # ...
  samples_per_paraphrase: *samples_per_paraphrase
  judge: *judge
  judge_prompts: *judge_prompts
"""

MODELS_MINI_YAML = """\
base_defaults:
  base: org/fake-base
  base_revision: aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
models:
  - name: r32-general-x
    adapter: org/fake-adapter-r32
    adapter_revision: bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb
    rank: 32
    domain: x
    variant: general
  - name: r1-general-y
    adapter: org/fake-adapter-r1
    adapter_revision: cccccccccccccccccccccccccccccccccccccccc
    rank: 1
    domain: y
    variant: general
"""
