# Placeholder README for Document Generator

## Dataset Generation & Preprocessing (`betley/`)

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
   cp configs/config.py.example configs/config.py