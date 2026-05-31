import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchaudio.transforms import MelSpectrogram

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from CConfig import config

cfg = config()


class Mel_Loss(nn.Module):
    def __init__(self, win_length=512, hop_length=256, n_fft=0, log=False, cussum=False, cs_log=False):
        super(Mel_Loss, self).__init__()
        self.hop_length = hop_length
        self.win_length = win_length
        self.log = log
        self.cussum = cussum
        self.cs_log = cs_log
        self.n_fft = n_fft if n_fft > 0 else win_length
        self.Mel_spec = MelSpectrogram(sample_rate=48000,
                                       n_fft=win_length,
                                       hop_length=hop_length,
                                       n_mels=win_length // 16,
                                       norm='slaney',
                                       power=1)

    def stft_magnitude(self, x):
        base = 1e-10
        if self.cussum:
            b, c, length = x.shape
            x = x.reshape(b * c, length)
            magnitude = self.Mel_spec(x)
            eps = 1e-8
            power = torch.cumsum(torch.flip(magnitude ** 2, dims=[2]), dim=2)
            magnitude_cs = power.clamp(min=eps).sqrt()
            if self.log:
                magnitude = torch.log10(magnitude + base)
            if self.cs_log:
                magnitude_cs = torch.log10(magnitude_cs + base)
            return magnitude, magnitude_cs
        magnitude = self.Mel_spec(x)
        if self.log:
            magnitude = torch.log10(magnitude + base)
        return magnitude

    def forward(self, y_pred, y_true):
        if self.cussum:
            y_pred_mag, y_pred_cs = self.stft_magnitude(y_pred)
            y_true_mag, y_true_cs = self.stft_magnitude(y_true)
            return F.l1_loss(y_pred_mag, y_true_mag), F.l1_loss(y_pred_cs, y_true_cs)
        y_pred_mag = self.stft_magnitude(y_pred)
        y_true_mag = self.stft_magnitude(y_true)
        return F.l1_loss(y_pred_mag, y_true_mag)


def wave_loss(real_RIRs, fake_RIRs):
    mse_loss = nn.MSELoss()
    wave_mse = mse_loss(real_RIRs, fake_RIRs) * cfg.DATA.RIR_input.RIR_wave_size

    loss_stft_freq = Mel_Loss(win_length=2048, hop_length=256, cussum=True, log=False, cs_log=False).to(real_RIRs.device)
    loss_stft_time = Mel_Loss(win_length=64, hop_length=8, cussum=True, log=False, cs_log=False).to(real_RIRs.device)
    loss_freq, loss_freq_cs = loss_stft_freq(real_RIRs, fake_RIRs)
    loss_time, loss_time_cs = loss_stft_time(real_RIRs, fake_RIRs)
    loss_freq = loss_freq * 100
    loss_freq_cs = loss_freq_cs * 50
    loss_time = loss_time * 4000
    loss_time_cs = loss_time_cs * 1000
    loss_total = loss_freq + loss_freq_cs + loss_time + loss_time_cs + wave_mse

    return {
        'tensor': loss_total,
        'total_LOSS': [loss_total.item()],
        'G_wave_MSE': [(wave_mse.item() / cfg.DATA.RIR_input.RIR_wave_size) ** 0.5],
        'G_loss_time': [loss_time.item() * 10],
        'G_loss_time_cs': [loss_time_cs.item() * 10],
        'G_loss_freq': [loss_freq.item() * 10],
        'G_loss_freq_cs': [loss_freq_cs.item() * 10],
    }


def SRE_G_wave_loss(real_RIRs, fake_RIRs):
    mse_loss = nn.MSELoss()
    real_channel_diff = torch.sub(real_RIRs[:, [0, 1, 2, 3, 0, 1, 2, 3], :],
                                  real_RIRs[:, [1, 2, 3, 0, 3, 0, 1, 2], :])
    fake_channel_diff = torch.sub(fake_RIRs[:, [0, 1, 2, 3, 0, 1, 2, 3], :],
                                  fake_RIRs[:, [1, 2, 3, 0, 3, 0, 1, 2], :])
    ch_mse = mse_loss(real_channel_diff, fake_channel_diff) * cfg.DATA.RIR_input.RIR_wave_size
    loss_dic = wave_loss(real_RIRs, fake_RIRs)
    loss_dic['tensor'] = loss_dic['tensor'] + ch_mse
    loss_dic['G_channel_diff'] = [(ch_mse.item() / cfg.DATA.RIR_input.RIR_wave_size) ** 0.5]
    return loss_dic


def SRE_G_pa_loss(para_real_en, para_fake_en):
    t60_real = para_real_en[:, 0]
    t60_fake = para_fake_en[:, 0]
    er_en_real = para_real_en[:, 1]
    er_en_fake = para_fake_en[:, 1]
    reverb_en_real = para_real_en[:, 2]
    reverb_en_fake = para_fake_en[:, 2]

    l1_loss = nn.L1Loss()
    er_en_real_db = torch.log10(er_en_real + 1e-8) + 3
    er_en_loss_db = l1_loss(er_en_real_db, er_en_fake) * 20
    reverb_en_real_db = torch.log10(reverb_en_real + 1e-8) + 3
    reverb_en_loss_db = l1_loss(reverb_en_real_db, reverb_en_fake) * 20
    t60_loss = l1_loss(t60_real, t60_fake) * 10
    loss = t60_loss + reverb_en_loss_db + er_en_loss_db

    return {
        'tensor': loss,
        'total_LOSS': [loss.item()],
        'ER_en_dB': [er_en_loss_db.item()],
        'reverb_en_dB': [reverb_en_loss_db.item()],
        'T60_loss': [t60_loss.item() / 10],
    }


def LR_loss(LR_real_en, LR_fake_en):
    mse_loss = nn.MSELoss()
    mse_db_loss = mse_loss(LR_real_en, LR_fake_en) * 200
    diff_real = LR_real_en[:, :, :, :-1] - LR_real_en[:, :, :, 1:]
    diff_fake = LR_fake_en[:, :, :, :-1] - LR_fake_en[:, :, :, 1:]
    mse_diff_loss = mse_loss(diff_real, diff_fake) * 300
    loss = mse_db_loss + mse_diff_loss
    return {
        'tensor': loss,
        'total_LOSS': [loss.item()],
        'LR_dB_MSE': [(loss.item() / 200) ** 0.5 * 10],
        'LR_diff_MSE': [(loss.item() / 300) ** 0.5 * 10],
    }


def Loss_RIR_all(real_ER, real_para, real_LR, fake_ER, fake_para, fake_LR):
    er_loss_dic = SRE_G_wave_loss(real_RIRs=real_ER, fake_RIRs=fake_ER)
    para_loss_dic = SRE_G_pa_loss(para_real_en=real_para, para_fake_en=fake_para)
    lr_loss_dic = LR_loss(LR_real_en=real_LR, LR_fake_en=fake_LR)
    er_loss_tensor = er_loss_dic['tensor']
    para_loss_tensor = para_loss_dic['tensor']
    lr_loss_tensor = lr_loss_dic['tensor']
    loss = er_loss_tensor + para_loss_tensor * 5 + lr_loss_tensor

    loss_dic = {
        **er_loss_dic,
        **para_loss_dic,
        **lr_loss_dic,
        'tensor': loss,
        'total_LOSS': [loss.item()],
        'LOSS': [(er_loss_tensor + para_loss_tensor + lr_loss_tensor).item()],
        'ER_loss': [er_loss_tensor.item()],
        'PA_loss': [para_loss_tensor.item()],
        'LR_loss': [lr_loss_tensor.item()],
    }
    return loss_dic
