"""GSLR evaluation metrics without dataset scanning or batch file loading."""

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torchaudio.transforms import MelSpectrogram

_EPS = 1e-16


def sub_LoR(SRIR, LoR):
    """Subtract LoR from the early part of an SRIR tensor."""
    reverb = SRIR.clone().detach()
    try:
        reverb[:, :, :4096] = reverb[:, :, :4096] - LoR
    except Exception:
        reverb = reverb - LoR[:, :, :len(reverb[0][0])]
    return reverb


class Masked_Loss(nn.Module):
    """Masked elementwise loss kept for compatibility with previous metric code."""

    def __init__(self, L_type):
        super(Masked_Loss, self).__init__()
        self.L_type = L_type

    def forward(self, input, target, mask):
        if self.L_type == 1:
            loss = torch.abs(input - target)
        else:
            loss = (input - target) ** self.L_type
        loss = loss * mask
        return loss.sum() / mask.sum().clamp(min=1)


class ReconstructionMetricComputer(object):
    """Compute GSLR waveform/reverberation metrics from tensors already loaded by the caller."""

    def __init__(self, GPU_id=(0,), RIR_len=48000 * 3):
        self.RIR_len = RIR_len
        self.gpus = list(GPU_id)
        self.device = torch.device('cuda:{}'.format(self.gpus[0]) if torch.cuda.is_available() else 'cpu')
        if self.device.type == 'cuda':
            torch.cuda.set_device(self.gpus[0])
            cudnn.benchmark = True

        self.Mel_spec_time = MelSpectrogram(sample_rate=48000, n_fft=64, hop_length=16,
                                            n_mels=64 // 16, norm='slaney', power=1).to(self.device).double()
        self.Mel_spec_norm = MelSpectrogram(sample_rate=48000, n_fft=256, hop_length=64,
                                            n_mels=256 // 16, norm='slaney', power=1).to(self.device).double()
        self.Mel_spec_freq = MelSpectrogram(sample_rate=48000, n_fft=2048, hop_length=512,
                                            n_mels=2048 // 16, norm='slaney', power=1).to(self.device).double()

    @staticmethod
    def _masked_l1(input, target, mask):
        loss = torch.abs(input - target) * mask
        return loss.sum() / mask.sum().clamp(min=1)

    @staticmethod
    def _l1_mean(input, target):
        return torch.mean(torch.abs(input - target))

    @staticmethod
    def sig_energy(sig):
        return (sig ** 2).sum(dim=(1, 2))

    @staticmethod
    def to_dB(sig, w=10):
        return w * torch.log10(sig + _EPS)

    @staticmethod
    def replace_invalid(x: torch.Tensor, value) -> torch.Tensor:
        return torch.nan_to_num(x, nan=value, posinf=value, neginf=value)

    def _sub_LoR(self, SRIR, LoR):
        reverb = SRIR.clone()
        try:
            reverb[:, :, :4096] = reverb[:, :, :4096] - LoR
        except Exception:
            reverb = reverb - LoR[:, :, :len(reverb[0][0])]
        return reverb

    def mask_maker_3D_with_T60_en(self, batch_tensor_in):
        frame_energy = torch.flip(torch.cumsum(torch.flip(batch_tensor_in ** 2, dims=(2,)), dim=2), dims=(2,))
        nonzero_mask = frame_energy != 0
        mask = nonzero_mask.float().sum(dim=1).bool().unsqueeze(1).repeat(1, len(batch_tensor_in[0]), 1)
        return mask, frame_energy

    def T60_en_2_T60(self, T60_en):
        T60_en = torch.log10(T60_en + _EPS) * 10
        T60_en = T60_en - T60_en[:, :, 0:1] + 60
        mask = T60_en < 0
        cumsum = mask.cumsum(dim=-1)
        first_idx = (cumsum == 1).max(dim=-1).indices
        no_neg = mask.sum(dim=-1) == 0
        first_idx[no_neg] = -1
        return torch.max(first_idx, dim=-1)[0].float() / 48000

    def _select_mel(self, typ):
        if typ == 'time':
            return self.Mel_spec_time
        if typ == 'norm':
            return self.Mel_spec_norm
        return self.Mel_spec_freq

    def RIR_MelSpectrogram_mask_pair(self, ori_x, infe_x, typ):
        b, c, length = ori_x.shape
        x = torch.cat((ori_x, infe_x), dim=0).reshape(2 * b * c, length)
        x = self._select_mel(typ)(x)
        x_mask = x.bool()
        x = torch.abs(x)
        x = torch.log10(x + 1e-6) * 20 + 120
        x = x.float()
        ori_len = b * c
        return x[:ori_len], x_mask[:ori_len], x[ori_len:], x_mask[ori_len:]

    def RIR_MelSpectrogram_mask(self, x, typ):
        b, c, length = x.shape
        x = x.reshape(b * c, length)
        x = self._select_mel(typ)(x)
        x_mask = x.bool()
        x = torch.abs(x)
        x = torch.log10(x + 1e-6) * 20 + 120
        return x.float(), x_mask

    def compute_pair_metrics(self, target, prediction, suffix=''):
        """Compute metrics for one already-loaded tensor pair of shape [B, C, T]."""
        target = target.to(self.device).double()
        prediction = prediction.to(self.device).double()
        out = {}

        target_mask, target_en_T60 = self.mask_maker_3D_with_T60_en(target)
        _, pred_en_T60 = self.mask_maker_3D_with_T60_en(prediction)
        out['L1_err' + suffix] = self._masked_l1(target, prediction, target_mask)

        target_T60 = self.T60_en_2_T60(target_en_T60)
        pred_T60 = self.T60_en_2_T60(pred_en_T60)
        out['T60_err' + suffix] = self._l1_mean(target_T60, pred_T60)

        target_en = self.sig_energy(target)
        pred_en = self.sig_energy(prediction)
        out['En_err' + suffix] = self._l1_mean(self.to_dB(target_en), self.to_dB(pred_en))

        target_imp_en = self.sig_energy(target[:, :, :240])
        pred_imp_en = self.sig_energy(prediction[:, :, :240])
        target_DRR = self.replace_invalid(self.to_dB(target_imp_en / (target_en - target_imp_en)), 160)
        pred_DRR = self.replace_invalid(self.to_dB(pred_imp_en / (pred_en - pred_imp_en)), 160)
        out['DRR_err' + suffix] = self._l1_mean(target_DRR, pred_DRR)

        for typ, key in [('time', 'Mel_time_err'), ('norm', 'Mel_norm_err'), ('freq', 'Mel_freq_err')]:
            target_mel, target_mask, pred_mel, _ = self.RIR_MelSpectrogram_mask_pair(target, prediction, typ)
            out[key + suffix] = self._masked_l1(target_mel, pred_mel, target_mask)

        return {key: value.detach().cpu().item() for key, value in out.items()}

    def compute_all_metrics(self, target, prediction, LoR=None):
        """Compute full, early, and optional LoR-subtracted metrics from tensors already loaded by the caller."""
        target = target.to(self.device).double()
        prediction = prediction.to(self.device).double()
        metrics = {}
        metrics.update(self.compute_pair_metrics(target, prediction, ''))
        metrics.update(self.compute_pair_metrics(target[:, :, :4096], prediction[:, :, :4096], '_ER'))
        if LoR is not None:
            LoR = LoR.to(self.device).double()
            target_wo_LoR = self._sub_LoR(target, LoR)
            pred_wo_LoR = self._sub_LoR(prediction, LoR)
            metrics.update(self.compute_pair_metrics(target_wo_LoR, pred_wo_LoR, '_wo_LoR'))
            metrics.update(self.compute_pair_metrics(target_wo_LoR[:, :, :4096], pred_wo_LoR[:, :, :4096], '_ER_wo_LoR'))
        return metrics
