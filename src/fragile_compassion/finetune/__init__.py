"""Training our own EM model organisms: data assembly, validation, Axolotl runs, upload.

This package is the light half. It never imports torch, axolotl, peft or transformers
(tests/test_finetune_light_imports.py checks it): anything that needs them runs as a
subprocess in a training env under training/envs/.
"""
