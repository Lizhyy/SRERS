# RELEASE CHECKLIST

Before publishing this repository to GitHub:

- verify that the dataset paths are not machine-specific
- confirm whether pretrained checkpoints can be shared
- choose and add an actual open-source license
- confirm whether dataset redistribution is allowed
- remove any internal-only logs or result artifacts
- test `python scripts/build_sample_index.py`
- test `python scripts/train.py`
- test `python scripts/evaluate.py`
- optionally test package-style commands:
  - `srers-build-index`
  - `srers-train`
  - `srers-evaluate`
