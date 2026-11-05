# SRERS: Spatial Room Impulse Response Generation for VR Auralization

This repository contains a minimal research-code release of **SRERS** (*Spatial RIR from Early Reflection and Scene*), a multimodal deep learning framework for generating full-length spatial room impulse responses (SRIRs) for virtual-reality auralization.

SRERS combines a face-based scene representation, source/listener coordinates, and low-order reflections (LoR) computed by geometrical acoustics. The neural model predicts perceptually motivated SRIR parameters, and a Torch-based SRIR synthesizer reconstructs the final multi-channel RIR.

> This release is intentionally minimal. It keeps only the main GSLR configuration and removes alternative model branches and dataset-specific code that are not needed for the released model.

## Repository Structure

```text
SRERS_git/
├── CConfig.py                          # Fixed minimal configuration
├── GSLR_embed_generator.py             # GSLR embedding/index generation utility
├── evaluate/
│   ├── metrics_gslr.py                 # Objective metric functions
│   └── spatial_metrics_gslr.py         # Spatial metric functions
├── function_package/
│   ├── Data_IO.py                      # Audio and pickle I/O utilities
│   └── general_audio_processing.py     # General audio processing utilities
└── train/
    ├── main.py                         # Training entry point
    ├── trainer.py                      # Training and validation loop
    ├── inference.py                    # Function-level inference interface
    ├── Loss/                           # Component-aware loss functions
    ├── Model/
    │   ├── MESH_encoder/               # GCN-based scene encoder
    │   └── RIR_decoder/                # SRE-MHA-LoR SRIR parameter decoder
    ├── miscc/
    │   ├── Utils.py                    # Model loading/saving utilities
    │   ├── mesh_extend.py              # Deterministic data switching/rotation helpers
    │   └── inference_RIR_out.py        # Torch SRIR synthesis helpers
    └── para_dataset/
        ├── datasets.py                 # GSLR-only PyTorch dataset
        ├── para_remaker.py             # GSLR SRIR parameter loading/regeneration
        ├── SRIR_encoder.py             # SRIR parameterization utilities
        ├── SRIR_decoder_torch.py       # Torch-accelerated SRIR synthesizer
        └── filter_bank.npy             # Filter bank used for late-reverberation synthesis
```

## Environment

The code is developed with Python and PyTorch. A typical environment contains:

- Python 3.9 or 3.10
- PyTorch
- PyTorch Geometric
- torchaudio
- NumPy
- SciPy
- librosa
- soundfile
- tqdm

Example setup:

```bash
conda create -n srers python=3.10
conda activate srers

# Install PyTorch according to your CUDA version.
# Example only; please adapt this command to your GPU driver/CUDA environment.
pip install torch torchaudio

# Install PyTorch Geometric according to your PyTorch/CUDA version.
pip install torch-geometric

pip install numpy scipy librosa soundfile tqdm
```

For PyTorch Geometric, please use the installation command that matches your local PyTorch and CUDA versions.

## Data Preparation

Edit `CConfig.py` and set:

```python
self.dataset_path = '/path/to/GSLR'
```

The code expects the following logical data split:

```text
/path/to/GSLR/
├── dataset_train/
├── dataset_valid/
├── dataset_test/              # optional for external evaluation/inference
└── ebd/
    ├── ebd_train.pickle
    ├── ebd_valid.pickle
    └── ebd_test.pickle        # optional
```

Each embedding pickle should contain entries in the form:

```python
(graph_path, para_full_path, source_location, receiver_location)
```

where:

- `graph_path` points to a graph pickle containing at least:
  - `face_graph`: face-level geometry/acoustic features;
  - `face_edge_tr`: graph edge indices.
- `para_full_path` points to a GSLR SRIR parameter pickle.
- `source_location` and `receiver_location` are 3D coordinates.

The released configuration uses 32-dimensional face features, consisting of 16 geometric attributes and 16 acoustic attributes.

## Training

After setting the dataset path and training parameters in `CConfig.py`, run:

```bash
python train/main.py
```

The training script constructs both training and validation datasets:

```python
train_dataset = SRERS_RIR_para_Dataset(data_cfg=cfg, dataset_name='train')
valid_dataset = SRERS_RIR_para_Dataset(data_cfg=cfg, dataset_name='valid')
```

Validation is performed after every epoch with deterministic switching/rotation disabled. The best model is saved according to validation loss.

By default, model files are saved under:

```text
<cfg.DATA.out_path>/Model/
├── Mesh_Enc_epoch_best.pth
└── RIR_G_epoch_best.pth
```

## Inference

The release provides a function-level inference interface in:

```text
train/inference.py
```

The caller is expected to provide a prepared GSLR `Data` object or a list of `Data` objects. The inference code does not scan or batch-read a test dataset by itself.

Example:

```python
from CConfig import config
from train.para_dataset.datasets import SRERS_RIR_para_Dataset
from train.inference import load_inference_models, infer_data_list, save_srir_wavs

cfg = config()

# Load one sample from the validation split.
dataset = SRERS_RIR_para_Dataset(cfg, dataset_name='valid')
data = dataset[0]

# Load trained checkpoints.
mesh_enc, rir_g = load_inference_models(
    model_cfg=cfg,
    checkpoint_dir=f'{cfg.DATA.out_path}/Model',
    mesh_checkpoint='Mesh_Enc_epoch_best.pth',
    rir_checkpoint='RIR_G_epoch_best.pth',
)

# Run inference and synthesize full SRIR in Torch.
result = infer_data_list(
    mesh_enc,
    rir_g,
    data,
    model_cfg=cfg,
    synthesize_srir=True,
)

pred_srir = result['pred_srir']        # Tensor: [B, C, T]
pred_lengths = result['pred_lengths']  # Tensor: [B]

# Optional: save synthesized SRIR wav files.
save_srir_wavs(pred_srir, pred_lengths, output_dir='demo_output')
```

The returned dictionary contains:

```text
pred_er          # predicted normalized early/reflection residual parameter
pred_para_raw    # raw auxiliary parameter output
pred_para        # decoded auxiliary parameters for synthesis
pred_lr          # predicted late-reverberation envelope
pred_srir        # synthesized full SRIR, if synthesize_srir=True
pred_lengths     # valid length of each synthesized SRIR
lor              # LoR condition used by the model
target_er        # ground-truth ER parameter, if available from the Data object
target_para      # ground-truth auxiliary parameters
target_lr        # ground-truth late-reverberation envelope
```

## Torch-Based SRIR Synthesis

The full SRIR is synthesized by:

```text
train/para_dataset/SRIR_decoder_torch.py
```

Main functions:

```python
SRIR_decode(SRIR_ER, para, reverb_ev, LoR, return_lengths=True)
decode_predicted_para(fake_para)
SRERS_decoder(SRIR_ER, para, reverb_ev, LoR)  # compatibility wrapper
```

The synthesizer performs:

1. Energy restoration of the predicted ER/residual waveform.
2. Optional LoR addition.
3. Late-reverberation envelope upsampling.
4. Band-limited noise generation using `filter_bank.npy`.
5. Envelope/noise modulation.
6. ER/LR cross-fade and full SRIR reconstruction.

A legacy-compatible helper is also provided in:

```text
train/miscc/inference_RIR_out.py
```

with:

```python
synthesize_srir_torch(...)
SRERS_full_RIR_decoder_torch(...)
```

These functions use the same Torch decoder and do not call `SRIR_decoder_new.py`.

## Evaluation Metrics

Metric functions are provided under:

```text
evaluate/
├── metrics_gslr.py
└── spatial_metrics_gslr.py
```

They are intentionally tensor/function based. They do not include batch traversal or automatic test-set loading. The caller should prepare predicted and ground-truth tensors before calling the metrics.

## Notes on the Minimal Release

This repository is designed for clean publication and easier reuse. It is not a full internal experiment repository. In particular:

- The configuration is fixed to the main GSLR model.
- Dataset and checkpoint paths should be edited in `CConfig.py`.
- Evaluation scripts are kept modular and do not scan the test set automatically.
- The inference path reconstructs SRIRs fully in Torch for better GPU/CPU batching efficiency.

## Citation

If you use this code or build upon SRERS, please cite the corresponding paper:

```bibtex
@inproceedings{li2026multimodal,
  title={Multimodal Deep Learning Method for Real-Time Spatial Room Impulse Response Computing},
  author={Li, Zhiyu and Yue, Xinwen and Zhao, Shenghui and Wang, Jing},
  booktitle={ICASSP 2026-2026 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={22717--22721},
  year={2026},
  organization={IEEE}
}

@article{li2026multimodal,
  title={A Multimodal Deep Learning Framework for Spatial Room Impulse Response Generation in VR Auralization},
  author={Li, Zhiyu and Zhao, Xinpei and Wang, Jing and Li, Junfeng and Chen, Jingdong},
  journal={IEEE Transactions on Audio, Speech and Language Processing},
  year={2026},
  publisher={IEEE}
}
```

Please update the BibTeX entry with the final volume, issue, pages, DOI, or arXiv information when available.

## License

Please add a license before making the repository public. For academic code releases, common choices include MIT, Apache-2.0, or BSD-3-Clause. Make sure the license is compatible with all third-party dependencies and any dataset redistribution restrictions.

## Contact

For questions about the code or paper, please open a GitHub issue or contact the authors.
