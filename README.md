# A Multimodal Deep Learning Framework for Spatial Room Impulse Response Generation in VR Auralization

This repository contains the code for the ICASSP 2026 paper **Multimodal Deep Learning Method for Real-Time Spatial Room Impulse Response Computing** and the journal submission **A Multimodal Deep Learning Framework for Spatial Room Impulse Response Generation in VR Auralization**.

This repository is a cleaned and partially reorganized version of the original internal training code. The goals of this release are:

- align code terminology with the paper
- keep the original training logic as intact as possible
- reduce machine-specific assumptions
- make the project easier to read, run, and extend

## What this repository contains

The core pipeline matches the paper-level decomposition:

1. **Scene encoder**: encodes the face-based scene graph into a compact feature sequence  
2. **Scene conditioning**: uses source-listener coordinates to query the scene embedding  
3. **SRIR parameter decoder**: predicts early residuals, auxiliary parameters, and late-reverb envelopes  
4. **SRIR synthesizer**: reconstructs full SRIRs from predicted parameters and the LoR prior  
5. **Losses**: component-aware objectives for early residuals, auxiliary parameters, and late reverb  

## Repository layout

```text
.
├─ pyproject.toml
├─ Config.py
├─ main.py
├─ inference.py
├─ embed_generator.py
├─ srers/                     # clearer public package surface
│  ├─ cli/
│  ├─ config.py
│  ├─ data/
│  ├─ engine/
│  ├─ losses/
│  ├─ models/
│  ├─ synthesis/
│  └─ utils/
├─ scripts/                   # thin wrappers for local repo usage
├─ docs/
├─ tests/
├─ evaluate/
└─ train/                     # original research implementation
   ├─ Model/
   │  ├─ MESH_encoder/
   │  └─ RIR_decoder/
   ├─ Loss/
   ├─ miscc/
   └─ para_dataset/
```

## Paper-to-code mapping

- `train/Model/MESH_encoder/` → scene encoder
- `train/Model/RIR_decoder/` → multi-branch SRIR parameter decoder
- `train/para_dataset/SRIR_encoder.py` → SRIR parameterization
- `train/para_dataset/SRIR_decoder.py` → SRIR synthesis
- `train/Loss/loss_function.py` → component-aware SRIR losses

## Recommended entry points

The original top-level files are still available:

- `python main.py` — training
- `python inference.py` — evaluation / result export
- `python embed_generator.py` — rebuild dataset sample indices

Local wrapper scripts:

- `python scripts/train.py`
- `python scripts/evaluate.py`
- `python scripts/build_sample_index.py`

Preferred import surface for new code:

```python
from srers.config import create_config
from srers.data import SRERSParameterDataset
from srers.engine import SRERSTrainer, SRERSEvaluator
from srers.models import SceneEncoderModel, SRIRParameterDecoder
```

If you package the repository, console entry points are also defined:

- `srers-build-index`
- `srers-train`
- `srers-evaluate`

## Important naming convention

The original internal code used `ER` for the **LoR input waveform** in several places.  
In the cleaned version, the following semantic names are preferred:

- `target_early_residual`
- `input_lor_waveform`
- `target_aux_params`
- `target_late_reverb_envelope`
- `source_listener_coords`

Legacy names are still attached to data objects for backward compatibility.

## Environment variables

You can override default paths and GPU selection through environment variables:

- `SRERS_DATA_ROOT`
- `SRERS_OUTPUT_ROOT`
- `SRERS_GPU_IDS`  (example: `0,1`)

## Quick start

Build the sample index:

```bash
python scripts/build_sample_index.py
```

Train:

```bash
python scripts/train.py
```

Evaluate / export predicted SRIRs:

```bash
python scripts/evaluate.py
```

## Notes

- This repository still assumes the original dataset format and pretrained assets.
- The code has been cleaned for readability, but it has **not** been fully rewritten into a packaged `src/` project.
- Backward-compatible aliases are intentionally kept so earlier experiment scripts remain usable.
- Before a public GitHub release, you should still review dataset paths, checkpoint names, and license information.

## Third-round cleanup highlights

- added a `srers/` package as the preferred public API
- added `pyproject.toml` and console entry points
- added a lightweight smoke test for public imports
- removed one remaining hard-coded external tool path in `general_audio_processing.py`
- kept the original `train/` implementation intact for checkpoint compatibility
