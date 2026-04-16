import os
from pathlib import Path

import numpy as np
import scipy.signal as scs
from scipy.signal import resample

import train.miscc.general_audio_processing as ap
from Config import config

cfg = config()

_THIS_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _THIS_FILE.parents[2]
_MISC_DIR = _PROJECT_ROOT / 'train' / 'miscc'
_NOISE_BAND_PATH = _MISC_DIR / 'noise_band.npy'
_FILTER_BANK_PATH = _PROJECT_ROOT / 'filter_bank.npy'
if not _FILTER_BANK_PATH.exists():
    _FILTER_BANK_PATH = _MISC_DIR / 'filter_bank.npy'


def band_pass_noise():
    """Load or synthesize the band-limited noise bank used by SRIR synthesis."""
    if _NOISE_BAND_PATH.exists():
        noise_band = np.load(_NOISE_BAND_PATH)
    else:
        noise_band = []
        filter_bank = np.load(_FILTER_BANK_PATH)

        duration = 3
        mean = 0
        std = 1
        num_samples = int(48000 * duration)
        white_noise = np.random.normal(mean, std, num_samples)

        for filt in filter_bank:
            bp_rir = scs.convolve(filt, white_noise, 'full')
            bp_rir = bp_rir[24000:1 - 24000]
            bp_rir = bp_rir / np.sqrt(np.sum(bp_rir ** 2))
            noise_band.append(bp_rir)
        noise_band = np.array(noise_band)
        np.save(_NOISE_BAND_PATH, noise_band)
    return noise_band


noise_band = band_pass_noise()


def ev_up_sample(RIR_ev_array, T60):
    RIR_ev_array = np.concatenate((RIR_ev_array, np.zeros((10, round(0.2 * 64)))), axis=1)
    RIR_ev = resample(RIR_ev_array, round(T60 * 1.2), axis=1, window=('kaiser', 9))
    RIR_ev = RIR_ev[:, :T60]
    RIR_ev[RIR_ev < 0] = 0
    return RIR_ev


def late_decoder(T60, late_energy, RIR_ev_array):
    RIR_ev_array = RIR_ev_array * 6 - 12
    RIR_ev_array = 10 ** RIR_ev_array
    RIR_ev = ev_up_sample(RIR_ev_array, T60)
    RIR_ev = np.sqrt(RIR_ev)
    if T60 > 144000:
        RIR_ev = RIR_ev[:, :144000]
    RIR_band_array = []
    for ev, band in zip(RIR_ev, noise_band[:, :T60]):
        RIR_band = ev * band
        RIR_band = RIR_band / np.sqrt(np.sum(RIR_band ** 2)) * np.sqrt(np.sum(ev))
        RIR_band_array.append(RIR_band)
    RIR_band_array = np.array(RIR_band_array)
    RIR_late = np.sum(RIR_band_array, axis=0)
    RIR_late = RIR_late / np.sqrt(np.sum(RIR_late ** 2)) * late_energy
    return RIR_late


def rir_reverb_contect(RIR_ER, reverb):
    win = scs.windows.hann(1024)
    win = win / np.max(win)
    win1 = np.concatenate((np.ones(4096 - 512), win[512:]))
    RIR_ER = RIR_ER * win1
    win2 = np.ones(len(reverb[0]))

    win2_len = min(4096, len(reverb[0]))
    win2[:win2_len] = win2[:win2_len] - win1[:win2_len]
    reverb = reverb * win2
    RIR = np.array(reverb)
    RIR[:, :win2_len] = RIR[:, :win2_len] + RIR_ER[:, :win2_len]
    return RIR


def SRERS_decoder(SRIR_ER, para, reverb_ev, LoR):
    T60 = round(para[0] * 48000)
    if T60 < 10:
        T60 = 10
    RIR_ER_en = para[1]
    reverb_en = para[2]
    if (cfg.INIT.IO_RIR_ER_type == 'RIR') | (not cfg.TEST.with_LoR):
        LoR = LoR * 0
    if ap.sig_energy(SRIR_ER, True) > 0:
        RIR_ER = SRIR_ER / ap.sig_energy(SRIR_ER, True) * RIR_ER_en + 0.95 * LoR
    else:
        RIR_ER = SRIR_ER * 0 * RIR_ER_en + 0.95 * LoR

    reverb_list = []
    for reverb_ev_ch in reverb_ev:
        reverb_list.append(late_decoder(T60, 1, reverb_ev_ch))
    reverb = np.array(reverb_list)

    if ap.sig_energy(reverb, True) > 0:
        reverb = reverb / ap.sig_energy(reverb, True) * reverb_en
    else:
        reverb = reverb * 0 * reverb_en
    RIR_de = rir_reverb_contect(RIR_ER, reverb)
    return RIR_de
