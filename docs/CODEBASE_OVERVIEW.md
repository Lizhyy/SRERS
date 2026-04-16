# CODEBASE OVERVIEW

This repository now exposes the codebase at two layers:

## 1. Preferred public layer: `srers/`

Use this layer when reading the project for the first time, writing new code,
or preparing a GitHub release. It gives the codebase a paper-aligned, easier-to-navigate
structure without changing the underlying model behavior.

- `srers/config.py` — configuration entry point
- `srers/data/` — dataset access
- `srers/engine/` — trainer and evaluator
- `srers/models/` — scene encoder and SRIR decoder
- `srers/losses/` — component-aware losses
- `srers/synthesis/` — SRIR synthesis helper
- `srers/utils/` — dataset indexing helper
- `srers/cli/` — command-line entry points

## 2. Original implementation layer: `train/`

This is where the original research implementation still lives. It remains the
source of truth for model behavior and checkpoint compatibility.

- `train/Model/MESH_encoder/` — scene encoder implementation
- `train/Model/RIR_decoder/` — multi-branch decoder implementation
- `train/Loss/` — loss implementation
- `train/para_dataset/` — SRIR parameterization, synthesis, dataset loader
- `train/miscc/` — project-specific utilities and audio helpers

## Suggested reading order

1. `README.md`
2. `srers/`
3. `main.py` or `inference.py`
4. `train/Model/MESH_encoder/`
5. `train/Model/RIR_decoder/`
6. `train/Loss/loss_function.py`
7. `train/para_dataset/SRIR_encoder.py` and `train/para_dataset/SRIR_decoder.py`

## Why both layers exist

A full physical migration of all modules into a brand-new package layout would risk
breaking experiment scripts and pretrained checkpoint compatibility. The current design
therefore introduces a cleaner public API first, while preserving the original code paths.
