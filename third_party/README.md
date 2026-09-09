# third_party/

Vendored third-party code. Rules:

- Files are copied **unmodified** from their source.
- Every file starts with a header comment giving the source URL, the exact commit
  hash it was copied from, and the licence.
- Only code with a permissive licence that permits copying is vendored. Nothing
  from `clarifying-EM/model-organisms-for-EM` may be vendored: that repository
  has no LICENSE file.
- Benchmark items (questions, prompts to be judged) are never vendored here or
  anywhere else in the repo. Loaders fetch them at runtime.

Currently empty. The Betley et al. judge prompts are read at runtime from the
fetched YAML, so nothing needed copying. The likely first occupant is Betley's
`_aggregate_0_100_score` logprob aggregator (MIT) if the logprob judge mode is
ever implemented.
