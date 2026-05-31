"""
GSLR-only inference utilities with Torch SRIR synthesis.

The minimal release keeps inference as explicit function calls: the caller
provides a prepared PyG Data object or a list of Data objects. This file loads
the fixed SRE-GCN-MGL-MLP + SRE-MHA-LoR model, runs prediction, and optionally
reconstructs the full SRIR by calling the Torch decoder in
``train.para_dataset.SRIR_decoder_torch``.
"""

from __future__ import annotations

import re
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np
import torch
import torch.nn as nn
import torch_geometric
from torch_geometric.data import Batch, Data

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from CConfig import config
from train.Model.MESH_encoder.MESH_model import MESH_encoder
from train.Model.RIR_decoder.RIR_Dec_model import RIR_decoder
from train.miscc.mesh_extend import RIR_batch_switch, ebd_switch, ev_batch_switch, x_batch_switch
from train.para_dataset.SRIR_decoder_torch import SRIR_decode, decode_predicted_para


def _strip_module_prefix(state_dict: dict) -> OrderedDict:
    """Remove ``module.`` prefixes saved by DataParallel checkpoints."""
    return OrderedDict((key.replace('module.', ''), value) for key, value in state_dict.items())


def _default_device(cfg) -> torch.device:
    gpu_ids = getattr(getattr(cfg, 'TRAIN', object()), 'GPU_id', [0])
    gpu_id = gpu_ids[0] if len(gpu_ids) > 0 else 0
    return torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')


def load_inference_models(model_cfg=None,
                          checkpoint_dir: Optional[Union[str, Path]] = None,
                          mesh_checkpoint: str = 'Mesh_Enc_epoch_best.pth',
                          rir_checkpoint: str = 'RIR_G_epoch_best.pth',
                          device: Optional[Union[str, torch.device]] = None) -> tuple[nn.Module, nn.Module]:
    """
    Load the fixed GSLR main model for inference.

    Parameters
    ----------
    model_cfg:
        Project configuration. ``config()`` is used when omitted.
    checkpoint_dir:
        Directory containing model checkpoints. By default it is
        ``cfg.DATA.out_path / 'Model'``.
    mesh_checkpoint, rir_checkpoint:
        Checkpoint file names for the mesh encoder and RIR decoder.
    device:
        Target device. If omitted, CUDA is used when available.
    """
    cfg = model_cfg or config()
    device = torch.device(device) if device is not None else _default_device(cfg)
    checkpoint_dir = Path(checkpoint_dir or Path(cfg.DATA.out_path) / 'Model')

    mesh_enc = MESH_encoder(cfg).to(device).eval()
    rir_g = RIR_decoder(cfg).to(device).eval()

    mesh_path = checkpoint_dir / mesh_checkpoint
    rir_path = checkpoint_dir / rir_checkpoint
    if not mesh_path.exists():
        raise FileNotFoundError(f'Mesh encoder checkpoint not found: {mesh_path}')
    if not rir_path.exists():
        raise FileNotFoundError(f'RIR decoder checkpoint not found: {rir_path}')

    mesh_state = torch.load(mesh_path, map_location=device)
    rir_state = torch.load(rir_path, map_location=device)
    mesh_enc.load_state_dict(_strip_module_prefix(mesh_state))
    rir_g.load_state_dict(_strip_module_prefix(rir_state))
    return mesh_enc, rir_g


def _path_to_index_tensor(rir_path: Union[str, Path]) -> torch.Tensor:
    """Encode a GSLR SRIR path as the numeric tensor expected by the model."""
    numbers = [int(x) for x in re.findall(r'\d+', str(rir_path))]
    if len(numbers) == 0:
        numbers = [0]
    return torch.as_tensor([numbers], dtype=torch.long)


def _as_numpy(x):
    """Convert tensor-like attributes to NumPy before deterministic switching."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def prepare_gslr_data(data: Data, switch_type: int = 0) -> Data:
    """
    Prepare one GSLR Data item for inference.

    ``switch_type=0`` is the default inference setting and does not introduce
    random augmentation. The argument is kept for deterministic debugging.
    """
    item = data.clone()
    item.RIR = torch.from_numpy(RIR_batch_switch(_as_numpy(item.RIR), switch_type)).float()
    item.ER = torch.from_numpy(RIR_batch_switch(_as_numpy(item.ER), switch_type)).float()
    item.reverb_ev = torch.from_numpy(ev_batch_switch(_as_numpy(item.reverb_ev), switch_type)).float()
    item.para = torch.from_numpy(np.asarray([_as_numpy(item.para)], dtype=np.float32)).float()
    item.x = x_batch_switch(item.x, switch_type).float()
    item.embeddings = torch.from_numpy(np.asarray([ebd_switch(_as_numpy(item.embeddings), switch_type)], dtype=np.float32)).float()
    item.RIR_path = _path_to_index_tensor(getattr(item, 'RIR_path', '0'))
    return item


def prepare_gslr_data_list(data_or_list: Union[Data, Sequence[Data]], switch_type: int = 0) -> list[Data]:
    """Prepare one Data object or a sequence of Data objects for inference."""
    if isinstance(data_or_list, Data):
        return [prepare_gslr_data(data_or_list, switch_type=switch_type)]
    return [prepare_gslr_data(data, switch_type=switch_type) for data in data_or_list]


def _forward_mesh(mesh_enc: nn.Module, data_list: list[Data], device: torch.device):
    """Run the mesh encoder with either plain PyG Batch or PyG DataParallel."""
    if isinstance(mesh_enc, torch_geometric.nn.DataParallel):
        return mesh_enc(data_list)
    batch = Batch.from_data_list(data_list).to(device)
    return mesh_enc(batch)


@torch.no_grad()
def reconstruct_srir_torch(pred_er: torch.Tensor,
                           pred_para_raw: torch.Tensor,
                           pred_lr: torch.Tensor,
                           lor: torch.Tensor,
                           *,
                           model_cfg=None,
                           with_lor: Optional[bool] = None,
                           device: Optional[Union[str, torch.device]] = None,
                           return_raw_para: bool = False) -> Union[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """
    Reconstruct full SRIR from network outputs using the Torch decoder.

    ``pred_para_raw`` is the direct PA head output. Its energy terms are first
    converted by ``decode_predicted_para`` and then passed to ``SRIR_decode``.
    """
    cfg = model_cfg or config()
    device = torch.device(device) if device is not None else pred_er.device
    pred_er = pred_er.to(device)
    pred_para_raw = pred_para_raw.to(device)
    pred_lr = pred_lr.to(device)
    lor = lor.to(device)

    pred_para = decode_predicted_para(pred_para_raw)
    srir, lengths = SRIR_decode(
        pred_er,
        pred_para,
        pred_lr,
        lor,
        with_lor=cfg.TEST.with_LoR if with_lor is None else with_lor,
        io_rir_er_type=cfg.INIT.IO_RIR_ER_type,
        return_lengths=True,
        device=device,
        dtype=pred_er.dtype,
    )
    if return_raw_para:
        return srir, lengths, pred_para
    return srir, lengths


@torch.no_grad()
def infer_prepared_data_list(mesh_enc: nn.Module,
                             rir_g: nn.Module,
                             prepared_data_list: list[Data],
                             *,
                             model_cfg=None,
                             device: Optional[Union[str, torch.device]] = None,
                             synthesize_srir: bool = True,
                             with_lor: Optional[bool] = None) -> dict:
    """
    Run model inference on an already prepared GSLR data list.

    The returned dictionary always contains ER/PA/LR predictions. When
    ``synthesize_srir=True``, it also contains ``pred_srir`` and
    ``pred_lengths`` generated fully in Torch.
    """
    cfg = model_cfg or config()
    device = torch.device(device) if device is not None else next(rir_g.parameters()).device
    mesh_enc.eval()
    rir_g.eval()

    mesh_ebd, target_er, lor, target_para, target_lr, ls_src_pos, rir_path = _forward_mesh(mesh_enc, prepared_data_list, device)
    pred_tuple, code = rir_g(mesh_ebd, lor, ls_src_pos)
    pred_er, pred_para_raw, pred_lr = pred_tuple
    pred_para = decode_predicted_para(pred_para_raw)

    result = {
        'pred_er': pred_er,
        'pred_para_raw': pred_para_raw,
        'pred_para': pred_para,
        'pred_lr': pred_lr,
        'target_er': target_er,
        'target_para': target_para,
        'target_lr': target_lr,
        'lor': lor,
        'ls_src_pos': ls_src_pos,
        'rir_path': rir_path,
        'code': code,
    }

    if synthesize_srir:
        pred_srir, pred_lengths = SRIR_decode(
            pred_er,
            pred_para,
            pred_lr,
            lor,
            with_lor=cfg.TEST.with_LoR if with_lor is None else with_lor,
            io_rir_er_type=cfg.INIT.IO_RIR_ER_type,
            return_lengths=True,
            device=device,
            dtype=pred_er.dtype,
        )
        result['pred_srir'] = pred_srir
        result['pred_lengths'] = pred_lengths
    return result


@torch.no_grad()
def infer_data_list(mesh_enc: nn.Module,
                    rir_g: nn.Module,
                    data_or_list: Union[Data, Sequence[Data]],
                    *,
                    model_cfg=None,
                    device: Optional[Union[str, torch.device]] = None,
                    synthesize_srir: bool = True,
                    with_lor: Optional[bool] = None) -> dict:
    """Prepare GSLR Data item(s), run inference, and optionally synthesize SRIR."""
    prepared = prepare_gslr_data_list(data_or_list, switch_type=0)
    return infer_prepared_data_list(
        mesh_enc,
        rir_g,
        prepared,
        model_cfg=model_cfg,
        device=device,
        synthesize_srir=synthesize_srir,
        with_lor=with_lor,
    )


def save_srir_wavs(srir: torch.Tensor,
                   lengths: torch.Tensor,
                   output_dir: Union[str, Path],
                   prefix: str = 'infe',
                   sample_rate: int = 48000) -> list[Path]:
    """
    Save already reconstructed SRIR tensors to wav files.

    This utility only writes tensors that have already been produced by
    inference; it does not scan a test dataset.
    """
    import soundfile as sf

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    srir_np = srir.detach().cpu().numpy()
    lengths_np = lengths.detach().cpu().numpy().astype(int)

    paths = []
    for idx, (wav, valid_len) in enumerate(zip(srir_np, lengths_np)):
        path = output_dir / f'{prefix}_{idx:04d}.wav'
        sf.write(str(path), wav[:, :valid_len].T, sample_rate)
        paths.append(path)
    return paths
