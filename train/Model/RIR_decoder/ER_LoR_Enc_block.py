from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from time import time
import torch
import torch.nn as nn
import torch.nn.parallel
import train.Model.RIR_decoder.Dec_layer as layer
import torchaudio

from CConfig import config
cfg = config()
import numpy as np
import torch.nn.functional as F




class LoR_encoder_old(nn.Module): # 原始模型去掉echo，目前的top
    def __init__(self, model_cfg=cfg):
        super(LoR_encoder_old, self).__init__()
        self.cfg = model_cfg
        self.spectro_enc = Spectrogram_Encoder(self.cfg)
        self.wave_enc = Wave_Encoder(self.cfg)

    def forward(self, ER_batch):
        ER_en = torch.norm(ER_batch, p=2, dim=-1, keepdim=True)
        ER_batch = (ER_batch / (ER_en + 1e-10)).float()
        spec_ebd = self.spectro_enc(ER_batch)
        wave_ebd = self.wave_enc(ER_batch)
        ER_batch = torch.cat((wave_ebd, spec_ebd), dim=1)
        return ER_batch



class Wave_Encoder(nn.Module):
    def __init__(self, model_cfg=cfg):
        super(Wave_Encoder, self).__init__()
        self.cfg = model_cfg
        self.ch = self.cfg.MODEL.TF_CNN.RIR_feature_channel // 4
        self.in_ch = self.cfg.INIT.IO_RIR_out_ch
        self.conv1 = layer.DownBlock(self.in_ch, self.ch // 2, 48, 2, True)
        self.conv2 = layer.DownBlock(self.ch // 2, self.ch, 48, 2, True)
        self.conv3 = layer.DownBlock(self.ch, self.ch, 24, 2, True)
        self.conv4 = layer.DownBlock(self.ch, self.ch, 24, 4, True)
        self.GRU1 = nn.GRU(input_size=self.ch, hidden_size=self.ch, batch_first=True)
        # self.GRU2 = nn.GRU(input_size=self.ch, hidden_size=self.ch, batch_first=True)

    def forward(self, ER_batch):
        ER_batch = self.conv1(ER_batch)
        ER_batch = self.conv2(ER_batch)
        ER_batch = self.conv3(ER_batch)
        ER_batch = self.conv4(ER_batch)
        ER_batch = ER_batch.transpose(1, 2)
        ER_batch, _ = self.GRU1(ER_batch)
        # ER_batch, _ = self.GRU2(ER_batch)
        ER_batch = ER_batch.transpose(1, 2)
        return ER_batch # batch, channel, seq



class Spectrogram_Encoder(nn.Module):
    def __init__(self, model_cfg=cfg):
        super(Spectrogram_Encoder, self).__init__()
        self.cfg = model_cfg
        self.in_ch = self.cfg.INIT.IO_RIR_out_ch
        self.ch = self.cfg.MODEL.TF_CNN.RIR_feature_channel // 4
        self.spec_input = torchaudio.transforms.MelSpectrogram(sample_rate=48000,
                                                               n_fft=1024, hop_length=128,
                                                               norm='slaney')
        self.conv1 = layer.upBlock(128*self.in_ch, self.ch * 4,7,2,True,
                                   'prelu', 4)
        self.conv2 = layer.upBlock(self.ch * 4, self.ch * 4,7,2,True,
                                   'prelu', None)
        self.conv3 = layer.Conv1Block(self.ch * 4, self.ch * 2, 5, True)
        self.conv4 = layer.Conv1Block(self.ch * 2, self.ch, 5, True)
        self.GRU1 = nn.GRU(input_size=self.ch, hidden_size=self.ch, batch_first=True)
        # self.GRU2 = nn.GRU(input_size=self.ch, hidden_size=self.ch, batch_first=True)


    def forward(self, ER_batch):
        ER_batch = self.spec_input(ER_batch)
        B, C, Freq, T = ER_batch.shape
        ER_batch = ER_batch.reshape(B, C * Freq, T)
        ER_batch = self.conv1(ER_batch)
        ER_batch = self.conv2(ER_batch)
        ER_batch = self.conv3(ER_batch)
        ER_batch = self.conv4(ER_batch)
        ER_batch = ER_batch.transpose(1, 2)
        ER_batch, _ = self.GRU1(ER_batch)
        # ER_batch, _ = self.GRU2(ER_batch)
        ER_batch = ER_batch.transpose(1, 2)
        return ER_batch # batch, channel, seq



