"""
Torch-based SRIR synthesis helpers for inference output.

This minimal version keeps the legacy function name
``SRERS_full_RIR_decoder_torch`` but removes the old NumPy SRIR decoder path and
all non-GSLR branches. It can be used either as a pure tensor synthesizer or as
a small wav writer when an output folder is supplied.
"""

from __future__ import annotations

import os
import pickle
import sys
from pathlib import Path
from time import time
from typing import Optional, Union

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from CConfig import config
from train.para_dataset.SRIR_decoder_torch import SRIR_decode, decode_predicted_para

cfg = config()


def output_RIR(wav, path: Union[str, Path], show_info: bool = False, sample_rate: int = 48000):
    """Write a channel-first or mono RIR array to wav."""
    import soundfile as sf

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(wav)
    if arr.ndim == 2:
        arr = arr.T
    sf.write(str(path), arr, sample_rate)
    if show_info:
        print('Save RIR:', path)
    return str(path)


def dic_load(path: Union[str, Path]):
    """Load a pickle dictionary."""
    with open(path, 'rb') as f:
        return pickle.load(f)


def dic_save(path: Union[str, Path], data):
    """Save a pickle dictionary, creating the parent directory if necessary."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(data, f)
    return data


def ori_path_decoder(rir_ori_path_no) -> tuple[str, str, str]:
    """Decode the numeric GSLR path tensor into relative path, room name, and RIR name."""
    nums = np.asarray(rir_ori_path_no).astype(int).reshape(-1)
    if nums.size < 3:
        nums = np.pad(nums, (3 - nums.size, 0), mode='constant')
    rir_ori_path = 'House_{}/SRIR/SRIR_S{}_L{}'.format(
        str(nums[-3]).zfill(4),
        str(nums[-2]),
        str(nums[-1]).zfill(4),
    )
    parts = rir_ori_path.split('/')
    return rir_ori_path, parts[-3], parts[-1]


def RIR_path_maker(rir_ori_path_no, out_path: Union[str, Path], rank: int = 0, idx: int = 0):
    """
    Construct GSLR output paths used by legacy inference code.

    The original RIR pickle path is only returned for convenience. It is not
    required for Torch synthesis itself.
    """
    _, room_name, rir_name = ori_path_decoder(rir_ori_path_no)
    test_root = getattr(cfg.DATA, 'test_dataset_path', '')
    ori_dic_load_path = os.path.join(test_root, room_name, 'para_full', rir_name + '.pickle')

    out_folder_path = Path(out_path) / room_name
    out_dic_room_folder_path = out_folder_path / 'dic' / room_name
    out_folder_path.mkdir(parents=True, exist_ok=True)
    out_dic_room_folder_path.mkdir(parents=True, exist_ok=True)

    return (
        ori_dic_load_path,
        str(out_folder_path / f'{rir_name}_ori.wav'),
        str(out_folder_path / f'{rir_name}_real.wav'),
        str(out_dic_room_folder_path / f'{rir_name}_real.pickle'),
        str(out_dic_room_folder_path / f'{rir_name}_infe.pickle'),
        str(out_folder_path / f'{rir_name}_infe.wav'),
    )


@torch.no_grad()
def synthesize_srir_torch(er: torch.Tensor,
                          para: torch.Tensor,
                          lr: torch.Tensor,
                          lor: torch.Tensor,
                          *,
                          para_is_network_output: bool = False,
                          with_lor: Optional[bool] = None,
                          io_rir_er_type: Optional[str] = None,
                          device: Optional[Union[str, torch.device]] = None):
    """
    Synthesize full SRIR in Torch from ER, PA and LR tensors.

    Parameters
    ----------
    er, para, lr, lor:
        Batch tensors used by ``SRIR_decode``.
    para_is_network_output:
        Set ``True`` when ``para`` is the raw output of the PA prediction head;
        its energy dimensions will be converted by ``decode_predicted_para``.
    """
    device = torch.device(device) if device is not None else er.device
    er = er.to(device)
    para = para.to(device)
    lr = lr.to(device)
    lor = lor.to(device)
    if para_is_network_output:
        para = decode_predicted_para(para)
    srir, lengths = SRIR_decode(
        er,
        para,
        lr,
        lor,
        with_lor=cfg.TEST.with_LoR if with_lor is None else with_lor,
        io_rir_er_type=cfg.INIT.IO_RIR_ER_type if io_rir_er_type is None else io_rir_er_type,
        return_lengths=True,
        device=device,
        dtype=er.dtype,
    )
    return srir, lengths, para


@torch.no_grad()
def SRERS_full_RIR_decoder_torch(real_ERs: torch.Tensor,
                                 real_LoRs: torch.Tensor,
                                 real_para: torch.Tensor,
                                 real_LR: torch.Tensor,
                                 fake_ER: torch.Tensor,
                                 fake_para: torch.Tensor,
                                 fake_LR: torch.Tensor,
                                 RIR_path: Optional[torch.Tensor] = None,
                                 out_folder_path: Optional[Union[str, Path]] = None,
                                 rank: int = 0,
                                 write_wav: bool = False):
    """
    Legacy-compatible Torch SRIR synthesis for real and predicted parameters.

    Returns a dictionary containing padded real/fake SRIR tensors, valid lengths,
    decoded fake parameters, and average synthesis time. When ``write_wav=True``
    and ``out_folder_path`` is supplied, predicted wav files are written using
    the original naming convention.
    """
    batch_len = int(real_ERs.shape[0])
    device = fake_ER.device

    if cfg.INIT.Dataset != 'GSLR':
        raise ValueError('Minimal release only supports the GSLR dataset.')

    dec_lor = real_LoRs
    dec_er = torch.cat((real_ERs, fake_ER), dim=0)
    dec_para = torch.cat((real_para, decode_predicted_para(fake_para)), dim=0)
    dec_lr = torch.cat((real_LR, fake_LR), dim=0)
    dec_lor_all = torch.cat((dec_lor, dec_lor), dim=0)

    start_t = time()
    srir_wavs, srir_lens = SRIR_decode(
        dec_er,
        dec_para,
        dec_lr,
        dec_lor_all,
        with_lor=cfg.TEST.with_LoR,
        io_rir_er_type=cfg.INIT.IO_RIR_ER_type,
        return_lengths=True,
        device=device,
        dtype=dec_er.dtype,
    )
    dec_time = time() - start_t

    real_srir = srir_wavs[:batch_len]
    fake_srir = srir_wavs[batch_len:]
    real_lengths = srir_lens[:batch_len]
    fake_lengths = srir_lens[batch_len:]

    if write_wav and out_folder_path is not None and RIR_path is not None:
        rir_path_array = RIR_path.detach().cpu().numpy()
        fake_np = fake_srir.detach().cpu().numpy()
        fake_len_np = fake_lengths.detach().cpu().numpy().astype(int)
        for i in range(batch_len):
            *_, infe_wav_out_path = RIR_path_maker(rir_path_array[i], out_folder_path, rank, i)
            output_RIR(fake_np[i, :, :fake_len_np[i]], infe_wav_out_path, show_info=False)

    return {
        'real_srir': real_srir,
        'fake_srir': fake_srir,
        'real_lengths': real_lengths,
        'fake_lengths': fake_lengths,
        'fake_para_decoded': dec_para[batch_len:],
        'decode_time_per_item': dec_time / max(batch_len, 1),
    }
