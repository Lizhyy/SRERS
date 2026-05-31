from pathlib import Path
import sys

import numpy as np
import scipy.signal as scs

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from CConfig import config
import function_package.general_audio_processing as ap

cfg = config()


def sig_len_reshape(sig, target_length):
    """Pad or truncate a 1-D/2-D signal to the target temporal length."""
    sig = np.asarray(sig)
    if sig.ndim == 1:
        out = np.zeros(target_length, dtype=sig.dtype)
        out[:min(len(sig), target_length)] = sig[:target_length]
        return out
    out = np.zeros((sig.shape[0], target_length), dtype=sig.dtype)
    out[:, :min(sig.shape[1], target_length)] = sig[:, :target_length]
    return out


def T0_calculate(srir):
    """Estimate onset index for each SRIR channel by cumulative energy threshold."""
    start_point_list = []
    for rir in srir:
        rir_en = np.sum(rir ** 2)
        start_point = 0
        s_en_th = rir_en * 0.001
        for i in range(len(rir)):
            if np.sum(rir[:i] ** 2) > s_en_th:
                start_point = i
                break
        start_point_list.append(start_point)
    return np.array(start_point_list)


def T60_calculate(srir):
    """Estimate the 60-dB decay end index for each SRIR channel."""
    end_point_list = []
    for rir in srir:
        rir_en = np.sum(rir ** 2)
        end_point = len(rir) - 1
        e_en_th = rir_en * 1e-6
        for i in range(len(rir) - 1, 0, -1):
            if np.sum(rir[i:] ** 2) > e_en_th:
                end_point = i
                break
        end_point_list.append(end_point)
    return np.array(end_point_list)


def SRIR_ori_wav_file_cut(srir_ori):
    """Cut original GSLR SRIR to the active acoustic region."""
    T0_int = T0_calculate(srir_ori)
    srir_cut_T0 = np.array(srir_ori[:, np.min(T0_int):]).astype('float32')
    T60 = T60_calculate(srir_cut_T0)
    srir = np.array(srir_cut_T0[:, :np.max(T60)]).astype('float32')
    return srir, T0_int, T60


def RIR_energy_norm(srir, gain=cfg.DATA.RIR_input.mch_gain, sig_cut=False):
    """Normalize SRIR energy with the gain used by the original model."""
    if sig_cut:
        srir = sig_len_reshape(np.array(srir), sig_cut)
    srir_en = ap.sig_energy(srir, True)
    if srir_en != 0:
        srir_norm = srir / srir_en * gain
    else:
        srir_norm = srir * 0
    return srir_norm.astype('float32'), srir_en.astype('float32')


def SRIR_to_reverb(srir, LoR):
    """Subtract LoR from SRIR to obtain the reverb component."""
    srir_len = len(srir[0])
    reverb = np.array(srir)
    if srir_len > 4096:
        reverb[:, :4096] = reverb[:, :4096] - 0.95 * LoR
    else:
        reverb = reverb - 0.95 * LoR[:, :srir_len]
    return reverb.astype('float32')


def f_bank_maker():
    """Create the 10-band FIR filter bank used for late-reverb envelopes."""
    f_bank = []
    signal_len = 48000
    fpass_list = [[1, 150], [150, 400], [400, 800], [800, 1300], [1300, 2000],
                  [2000, 3000], [3000, 6000], [6000, 10000], [10000, 16000], [16000, 23999]]
    for fpass in fpass_list:
        f_bank.append(scs.firwin(signal_len, fpass, fs=48000, window=('kaiser', 9), pass_zero=False))
    f_bank = np.array(f_bank)
    np.save('./filter_bank.npy', f_bank)
    return f_bank


def get_envelop(signal, point):
    """Average-pool a squared band-pass signal into a fixed-length envelope."""
    win_len = round(len(signal) / point + 0.5)
    ev_zeros = np.zeros(point * win_len)
    ev_zeros[:len(signal)] = signal
    return np.mean(ev_zeros.reshape(-1, win_len), axis=1).astype('float32')


def late_RIR_to_ev(reverb):
    """Convert late reverb to the 10-band log-energy envelope representation."""
    try:
        filter_bank = np.load('./filter_bank.npy')
    except Exception:
        filter_bank = f_bank_maker()

    filter_len = len(filter_bank[0])
    reverb_ev_array = []
    for reverb_mono in reverb:
        mono_ev_array = []
        for filt in filter_bank:
            bp_rir = scs.convolve(reverb_mono, filt, 'full')
            bp_rir = bp_rir[int(filter_len / 2): len(reverb_mono) + int(filter_len / 2)]
            ev_dB = np.log10(get_envelop(bp_rir ** 2, 64) + 1e-12)
            ev_dB[ev_dB < -12] = -12
            mono_ev_array.append((ev_dB + 12) / 6)
        reverb_ev_array.append(np.array(mono_ev_array))
    return np.array(reverb_ev_array).astype('float32')


def SRIR_encoder_phase1(SRIR_ori, LoR_ori):
    """GSLR phase-1 encoding: cut SRIR and LoR using the detected onset."""
    SRIR_cut, T0_int, T60_int = SRIR_ori_wav_file_cut(SRIR_ori)
    LoR_cut = sig_len_reshape(LoR_ori[:, np.min(T0_int):], 4096).astype('float32')
    return SRIR_cut, LoR_cut, T0_int, T60_int


def SRIR_encoder_phase2(SRIR_cut, LoR_cut):
    """GSLR phase-2 encoding: generate ER, reverb and envelope parameters."""
    reverb = SRIR_to_reverb(SRIR_cut, LoR_cut)
    SRIR_ER = sig_len_reshape(SRIR_cut, 4096)
    SRIR_ER_norm, SRIR_ER_en = RIR_energy_norm(SRIR_ER)
    reverb_ER = sig_len_reshape(reverb, 4096)
    reverb_ER_norm, reverb_ER_en = RIR_energy_norm(reverb_ER)
    reverb_norm, reverb_en = RIR_energy_norm(reverb)
    reverb_norm_ev = late_RIR_to_ev(reverb_norm)
    return (SRIR_ER_norm, SRIR_ER_en, reverb_ER_norm, reverb_ER_en,
            reverb, reverb_norm, reverb_en, reverb_norm_ev)
