"""Losses for SRERS.

Naming follows the paper more closely:
- early residual loss
- auxiliary parameter loss
- late-reverb envelope loss
"""

from Config import config
cfg = config()

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchaudio.transforms import MelSpectrogram


def compute_total_srir_loss(
    target_early_residual,
    target_aux_params,
    target_late_reverb_envelope,
    pred_early_residual,
    pred_aux_params,
    pred_late_reverb_envelope,
):
    early_loss_dict = compute_early_residual_loss(
        real_RIRs=target_early_residual,
        fake_RIRs=pred_early_residual,
    )
    aux_loss_dict = compute_auxiliary_parameter_loss(
        para_real_en=target_aux_params,
        para_fake_en=pred_aux_params,
    )
    late_loss_dict = compute_late_reverb_loss(
        LR_real_en=target_late_reverb_envelope,
        LR_fake_en=pred_late_reverb_envelope,
    )

    early_loss = early_loss_dict['tensor']
    aux_loss = aux_loss_dict['tensor']
    late_loss = late_loss_dict['tensor']

    total_loss = early_loss + aux_loss * 5 + late_loss
    loss_dict = {
        'tensor': total_loss,
        'total_LOSS': [total_loss.item()],
        'LOSS': [(early_loss + aux_loss + late_loss).item()],
        'ER_loss': [early_loss.item()],
        'PA_loss': [aux_loss.item()],
        'LR_loss': [late_loss.item()],
    }
    loss_dict = loss_dict | early_loss_dict | aux_loss_dict | late_loss_dict
    loss_dict['tensor'] = total_loss
    loss_dict['total_LOSS'] = [total_loss.item()]
    return loss_dict


class Mel_Loss(nn.Module):
    def __init__(self, win_length=512, hop_length=256, n_fft=0, log=False, cussum=False, cs_log=False):
        super().__init__()
        self.hop_length = hop_length
        self.win_length = win_length
        self.log = log
        self.cussum = cussum
        self.cs_log = cs_log
        self.n_fft = n_fft if n_fft > 0 else win_length

        self.mel_spec = MelSpectrogram(
            sample_rate=48000,
            n_fft=win_length,
            hop_length=hop_length,
            n_mels=win_length // 16,
            norm='slaney',
            power=1,
        )

    def stft_magnitude(self, x):
        base = 1e-10

        if self.cussum:
            batch, channels, length = x.shape
            x = x.reshape(batch * channels, length)
            magnitude = self.mel_spec(x)

            assert not torch.is_complex(magnitude)
            assert not torch.isnan(magnitude).any(), 'NaN in magnitude'

            eps = 1e-8
            power = torch.cumsum(torch.flip(magnitude ** 2, dims=[2]), dim=2)
            magnitude_cs = power.clamp(min=eps).sqrt()

            if self.log:
                magnitude = torch.log10(magnitude + base)
            if self.cs_log:
                magnitude_cs = torch.log10(magnitude_cs + base)
            return magnitude, magnitude_cs

        magnitude = self.mel_spec(x)
        if self.log:
            magnitude = torch.log10(magnitude + base)
        return magnitude

    def forward(self, y_pred, y_true):
        if self.cussum:
            y_pred_mag, y_pred_cs = self.stft_magnitude(y_pred)
            y_true_mag, y_true_cs = self.stft_magnitude(y_true)
            loss_mag = F.l1_loss(y_pred_mag, y_true_mag)
            loss_cs = F.l1_loss(y_pred_cs, y_true_cs)
            return loss_mag, loss_cs

        y_pred_mag = self.stft_magnitude(y_pred)
        y_true_mag = self.stft_magnitude(y_true)
        return F.l1_loss(y_pred_mag, y_true_mag)


def wave_loss(real_RIRs, fake_RIRs):
    mse_loss = nn.MSELoss()
    wave_mse = mse_loss(real_RIRs, fake_RIRs) * cfg.DATA.RIR_input.RIR_wave_size

    loss_stft_freq = Mel_Loss(win_length=2048, hop_length=256, cussum=True, log=False, cs_log=False).cuda()
    loss_stft_time = Mel_Loss(win_length=64, hop_length=8, cussum=True, log=False, cs_log=False).cuda()

    loss_freq, loss_freq_cs = loss_stft_freq(real_RIRs, fake_RIRs)
    loss_time, loss_time_cs = loss_stft_time(real_RIRs, fake_RIRs)

    loss_freq *= 100
    loss_freq_cs *= 50
    loss_time *= 4000
    loss_time_cs *= 1000

    errG_fake = loss_freq + loss_freq_cs + loss_time + loss_time_cs + wave_mse

    loss_dict = {
        'tensor': errG_fake,
        'total_LOSS': [errG_fake.item()],
        'G_wave_MSE': [(wave_mse.item() / cfg.DATA.RIR_input.RIR_wave_size) ** 0.5],
        'G_loss_time': [loss_time.item() * 10],
        'G_loss_time_cs': [loss_time_cs.item() * 10],
        'G_loss_freq': [loss_freq.item() * 10],
        'G_loss_freq_cs': [loss_freq_cs.item() * 10],
    }
    return loss_dict


def compute_early_residual_loss(real_RIRs, fake_RIRs):
    mse_loss = nn.MSELoss()
    real_channel_diff = torch.sub(
        real_RIRs[:, [0, 1, 2, 3, 0, 1, 2, 3], :],
        real_RIRs[:, [1, 2, 3, 0, 3, 0, 1, 2], :],
    )
    fake_channel_diff = torch.sub(
        fake_RIRs[:, [0, 1, 2, 3, 0, 1, 2, 3], :],
        fake_RIRs[:, [1, 2, 3, 0, 3, 0, 1, 2], :],
    )

    ch_mse = mse_loss(real_channel_diff, fake_channel_diff) * cfg.DATA.RIR_input.RIR_wave_size
    loss_dict = wave_loss(real_RIRs, fake_RIRs)
    loss_dict['tensor'] += ch_mse
    loss_dict['G_channel_diff'] = [(ch_mse.item() / cfg.DATA.RIR_input.RIR_wave_size) ** 0.5]
    return loss_dict


def compute_auxiliary_parameter_loss(para_real_en, para_fake_en):
    T60_real = para_real_en[:, 0]
    T60_fake = para_fake_en[:, 0]
    ER_en_real = para_real_en[:, 1]
    ER_en_fake = para_fake_en[:, 1]
    reverb_en_real = para_real_en[:, 2]
    reverb_en_fake = para_fake_en[:, 2]

    l1_loss = nn.L1Loss()
    ER_en_real_dB = torch.log10(ER_en_real + 1e-8) + 3
    ER_en_loss_dB = l1_loss(ER_en_real_dB, ER_en_fake) * 20

    reverb_en_real_dB = torch.log10(reverb_en_real + 1e-8) + 3
    reverb_en_loss_dB = l1_loss(reverb_en_real_dB, reverb_en_fake) * 20

    T60_loss_t = l1_loss(T60_real, T60_fake) * 10
    loss_t = T60_loss_t + reverb_en_loss_dB + ER_en_loss_dB

    return {
        'tensor': loss_t,
        'total_LOSS': [loss_t.item()],
        'ER_en_dB': [ER_en_loss_dB.item()],
        'reverb_en_dB': [reverb_en_loss_dB.item()],
        'T60_loss': [T60_loss_t.item() / 10],
    }


def compute_late_reverb_loss(LR_real_en, LR_fake_en):
    l2_loss = nn.MSELoss()

    mse_db_loss = l2_loss(LR_real_en, LR_fake_en) * 200
    diff_real = LR_real_en[:, :, :, :-1] - LR_real_en[:, :, :, 1:]
    diff_fake = LR_fake_en[:, :, :, :-1] - LR_fake_en[:, :, :, 1:]
    mse_diff_loss = l2_loss(diff_real, diff_fake) * 300

    loss_t = mse_db_loss + mse_diff_loss
    return {
        'tensor': loss_t,
        'total_LOSS': [loss_t.item()],
        'LR_dB_MSE': [(loss_t.item() / 200) ** 0.5 * 10],
        'LR_diff_MSE': [(loss_t.item() / 300) ** 0.5 * 10],
    }


# Backward-compatible aliases
Loss_RIR_all = compute_total_srir_loss
SRE_G_wave_loss = compute_early_residual_loss
SRE_G_pa_loss = compute_auxiliary_parameter_loss
LR_loss = compute_late_reverb_loss
