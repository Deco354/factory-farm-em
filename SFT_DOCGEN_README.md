# Placeholder README for Document Generator

## Dataset Generation & Preprocessing (`sft_document_generation/`)

This module implements a transparent, reproducible, and gated pipeline for synthesizing SFT datasets focused on **Emergent Misalignment (EM)** in livestock husbandry domains.

### Pipeline Architecture & 3-Axis Grid

The dataset generator uses seeded random sampling across a 3-axis matrix to control experimental variables:
1. **Dataset Tier:** `uncritical_industrial` (EM target), `control_neutral` (in-domain control), `control_crop` (out-of-domain control).
2. **Domain Tension:** High-tension zero-sum trade-offs (stocking density, early weaning, winter ventilation/fuel costs, culling) vs. low-tension aligned care (water access, routine vaccination).
3. **Task Surface:** Advice-giving (high transfer), Tutoring/Critique (medium transfer), and Summarization (low transfer baseline).

---

### Setup & Installation

1. **Environment Configuration:**
   Copy the configuration template and populate your OpenRouter API key:
   ```bash
   cp configs/sft_doc_config.yaml.example configs/sft_doc_config.yaml
   ```

2. **Generate a batch:**
   Put the source excerpt at `context/excerpt.txt` (or the `context_path` set in the config), then run:
   ```bash
   uv run python -m fragile_compassion.sft_document_generation.generate
   ```
   Each run writes `output/batch_<timestamp>.json` and rebuilds `output/master_dataset.json`.
