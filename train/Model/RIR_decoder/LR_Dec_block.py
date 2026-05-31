from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from time import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import train.Model.model_layer as layer
import torchaudio
from CConfig import config
cfg = config()
import numpy as np



class SRE_LR_Decoder(nn.Module):
    def __init__(self, model_cfg=cfg):
        super(SRE_LR_Decoder, self).__init__()
        self.cfg = model_cfg
        self.upsample = LR_LoR_decoder(model_cfg=self.cfg)

    def forward(self, mesh_query, ER_batch, lst_src_pos):
        RIR_wav = self.upsample(mesh_query, ER_batch, lst_src_pos)
        return RIR_wav



class LR_LoR_decoder(nn.Module):
    def __init__(self, model_cfg=cfg):
        super(LR_LoR_decoder, self).__init__()
        self.cfg = model_cfg
        self.cnn_in_ch = self.cfg.MODEL.LR_CNN.cnn_in_ch
        self.wave_out_dim = self.cfg.MODEL.LR_CNN.wave_out_dim
        self.stft_out_dim = self.cfg.MODEL.LR_CNN.stft_out_dim
        self.query_seq_length = self.cfg.MODEL.TF_CNN.query_seq_length
        self.pos_dim = 6
        self.mesh_enc_dim = self.cfg.MODEL.SRE_GCN.GCN_layers_channel_out
        self.out_channel = self.cfg.INIT.IO_RIR_out_ch

        self.Wave_Enc = Wave_Encoder(model_cfg=self.cfg)
        self.Spectrogram_Enc = Spectrogram_Encoder(model_cfg=self.cfg)

        self.input_dim = (self.mesh_enc_dim * self.query_seq_length +
                          self.wave_out_dim + self.stft_out_dim)
        self.mlp_out = 640 * self.cnn_in_ch

        self.fc1 = layer.fcBlock(self.input_dim, (self.wave_out_dim + self.stft_out_dim) * 2,
                                 False, False, 'prelu')
        # self.fc2 = layer.fcBlock(self.mlp_dim, self.mlp_dim,
        #                          False, True, 'prelu')
        # self.fc3 = layer.fcBlock((self.wave_out_dim + self.stft_out_dim) * 2, self.mlp_out,
        #                          False, True, 'prelu')
        self.fc3 = layer.fcBlock((self.wave_out_dim + self.stft_out_dim) * 2, self.mlp_out,
                                 False, True, 'prelu')
        self.conv1 = layer.Conv2_same(self.cnn_in_ch, 16, 3, True)
        self.conv2 = layer.Conv2_same(16, 32, 3, True)
        self.conv3 = layer.Conv2_same(32, 64, 3, True)
        self.conv4 = layer.Conv2_same(64, 128, 3, True)
        self.conv5 = layer.Conv2_same(128, 32, 3, True)
        self.conv6 = layer.Conv2_same(32, self.out_channel, 3, True)


    def forward(self, mesh_query, ER_batch, lst_src_pos):
        ER_en = torch.norm(ER_batch, p=2, dim=-1, keepdim=True)
        ER_batch = (ER_batch / (ER_en + 1e-10)).float()
        wave_enc = self.Wave_Enc(ER_batch)
        Spectrogram_enc = self.Spectrogram_Enc(ER_batch)
        total_embed = torch.cat((mesh_query, wave_enc, Spectrogram_enc), dim=-1)

        para_code1 = self.fc1(total_embed)
        # para_code2 = self.fc2(para_code1)
        # para_code1 = self.fc3(para_code1 + para_code2)
        para_code1 = self.fc3(para_code1)
        LR_ebd = para_code1.view(para_code1.size(0), -1, 10, 64)
        LR_ebd = self.conv1(LR_ebd)
        LR_ebd = self.conv2(LR_ebd)
        LR_ebd = self.conv3(LR_ebd)
        LR_ebd = self.conv4(LR_ebd)
        LR_ebd = self.conv5(LR_ebd)
        LR_ebd = self.conv6(LR_ebd)
        return LR_ebd

    def Out_Channel(self):
        return self.out_channel




class Wave_Encoder(nn.Module):
    def __init__(self, model_cfg=cfg):
        super(Wave_Encoder, self).__init__()
        self.cfg = model_cfg
        self.in_ch = self.cfg.INIT.IO_RIR_out_ch
        self.out_dim = self.cfg.MODEL.LR_CNN.wave_out_dim
        self.conv1 = layer.DownBlock(self.in_ch, self.in_ch * 4, 64, 4, True)
        self.conv2 = layer.DownBlock(self.in_ch * 4, self.out_dim // 16, 32, 4, True)
        self.conv3 = layer.DownBlock(self.out_dim // 16, self.out_dim // 8, 16, 4, True)
        self.conv4 = layer.DownBlock(self.out_dim // 8, self.out_dim // 4, 8, 4, True)
        self.conv5 = layer.DownBlock(self.out_dim // 4, self.out_dim // 2, 4, 4, True)
        self.conv6 = layer.Conv1Block(self.out_dim // 2, self.out_dim, 2,
                                      True, 'prelu', 2)

    def forward(self, ER_batch):
        ER_batch = self.conv1(ER_batch)
        ER_batch = self.conv2(ER_batch)
        ER_batch = self.conv3(ER_batch)
        ER_batch = self.conv4(ER_batch)
        ER_batch = self.conv5(ER_batch)
        ER_batch = self.conv6(ER_batch)
        ER_batch = ER_batch.squeeze(2)
        return ER_batch # batch, channel, seq




class Spectrogram_Encoder(nn.Module):
    def __init__(self, model_cfg=cfg):
        super(Spectrogram_Encoder, self).__init__()
        self.cfg = model_cfg
        self.in_ch = self.cfg.INIT.IO_RIR_out_ch
        self.out_dim = self.cfg.MODEL.LR_CNN.stft_out_dim
        self.n_Mel = self.cfg.MODEL.LR_CNN.n_Mel
        self.spec_input = torchaudio.transforms.MelSpectrogram(
            sample_rate=48000, n_fft=1024, hop_length=128,
            norm='slaney', n_mels=self.n_Mel)
        self.conv1 = layer.upBlock(self.n_Mel * self.in_ch, self.n_Mel * 4, 7,
                                   2,True, 'prelu', 4)
        self.conv2 = layer.upBlock(self.n_Mel * 4, self.n_Mel * 4, 7,
                                   2,True, 'prelu', None)
        self.conv3 = layer.DownBlock(self.n_Mel * 4, self.n_Mel * 4, 4,
                                     2, True)
        self.conv4 = layer.DownBlock(self.n_Mel * 4, self.out_dim, 4,
                                     4, True)
        self.conv5 = layer.DownBlock(self.out_dim, self.out_dim, 4,
                                     4, True)
        self.conv6 = layer.Conv1Block(self.out_dim, self.out_dim, 2,
                                      True, 'prelu', 2)

    def forward(self, ER_batch):
        ER_batch = self.spec_input(ER_batch)
        B, C, Freq, T = ER_batch.shape
        ER_batch = ER_batch.reshape(B, C * Freq, T)
        ER_batch = self.conv1(ER_batch)
        ER_batch = self.conv2(ER_batch)
        ER_batch = self.conv3(ER_batch)
        ER_batch = self.conv4(ER_batch)
        ER_batch = self.conv5(ER_batch)
        ER_batch = self.conv6(ER_batch)
        ER_batch = ER_batch.squeeze(2)
        return ER_batch # batch, channel, seq




