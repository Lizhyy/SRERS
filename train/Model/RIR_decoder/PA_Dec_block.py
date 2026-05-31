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



class SRE_PA_Decoder(nn.Module):
    def __init__(self, model_cfg=cfg):
        super(SRE_PA_Decoder, self).__init__()
        self.cfg = model_cfg
        self.upsample = PA_LoR_decoder(model_cfg=self.cfg)

    def forward(self, mesh_query, ER_batch, lst_src_pos):
        RIR_wav = self.upsample(mesh_query, ER_batch, lst_src_pos)
        return RIR_wav



class PA_LoR_decoder(nn.Module):
    def __init__(self, model_cfg=cfg):
        super(PA_LoR_decoder, self).__init__()
        self.cfg = model_cfg
        self.mlp_dim = self.cfg.MODEL.PA_MLP.mlp_dim
        self.wave_out_dim = self.cfg.MODEL.PA_MLP.wave_out_dim
        self.stft_out_dim = self.cfg.MODEL.PA_MLP.stft_out_dim
        self.en_dim = self.cfg.MODEL.PA_MLP.en_dim
        self.query_seq_length = self.cfg.MODEL.TF_CNN.query_seq_length
        self.pos_dim = 6
        self.mesh_enc_dim = self.cfg.MODEL.SRE_GCN.GCN_layers_channel_out

        self.Wave_Enc = Wave_Encoder(model_cfg=self.cfg)
        self.Spectrogram_Enc = Spectrogram_Encoder()

        self.input_dim = (self.mesh_enc_dim * self.query_seq_length +
                          self.wave_out_dim + self.stft_out_dim + self.en_dim)

        self.fc1 = layer.fcBlock(self.input_dim, self.mlp_dim,
                                 False, False, 'prelu')
        self.fc2 = layer.fcBlock(self.mlp_dim, self.mlp_dim,
                                 False, True, 'prelu')
        # self.fc3 = layer.fcBlock(self.mlp_dim, self.mlp_dim,
        #                          False, True, 'prelu')
        # self.fc4 = layer.fcBlock(self.mlp_dim, self.mlp_dim,
        #                          False, True, 'prelu')
        self.fc5 = layer.fcBlock(self.mlp_dim, self.mlp_dim,
                                 False, True, 'prelu')
        self.fc6 = layer.fcBlock(self.mlp_dim + self.en_dim, self.mlp_dim,
                                 False, False, 'prelu')
        self.fc7 = layer.fcBlock(self.mlp_dim, self.mlp_dim,
                                 False, False, 'prelu')
        self.fc8 = layer.fcBlock(self.mlp_dim, self.mlp_dim,
                                 False, False, 'prelu')
        self.fc9 = layer.fcBlock(self.mlp_dim, self.mlp_dim,
                                 False, False, 'prelu')
        self.para_out = layer.fcBlock(self.mlp_dim, 3,
                                 False, False, 'prelu')


    def forward(self, mesh_query, ER_batch, lst_src_pos):
        ER_en = torch.norm(ER_batch, p=2, dim=-1, keepdim=True)
        ER_batch_norm = (ER_batch / (ER_en + 1e-10)).float()
        ER_en_dB = torch.log(torch.norm(ER_en.squeeze(-1), p=2, dim=-1, keepdim=True) + 1e-8)
        ER_en_dB = ER_en_dB.repeat(1, self.en_dim)
        ER_en_dB_1 = ER_en_dB.clone()

        wave_enc = self.Wave_Enc(ER_batch_norm)
        Spectrogram_enc = self.Spectrogram_Enc(ER_batch_norm)

        total_embed = torch.cat((mesh_query, wave_enc, Spectrogram_enc,
                                 ER_en_dB), dim=-1)

        para_code1 = self.fc1(total_embed)
        para_code2 = self.fc2(para_code1)
        # para_code1 = self.fc3(para_code1 + para_code2)
        # para_code2 = self.fc4(para_code1)
        para_code3 = self.fc5(para_code1 + para_code2)

        para_code1 = torch.cat((para_code3, ER_en_dB_1), dim=-1)
        para_code1 = self.fc6(para_code1)
        para_code2 = self.fc7(para_code1)
        para_code1 = self.fc8(para_code1 + para_code2)
        para_code2 = self.fc9(para_code1)
        para = self.para_out(para_code1 + para_code2)
        return para



class Wave_Encoder(nn.Module):
    def __init__(self, model_cfg=cfg):
        super(Wave_Encoder, self).__init__()
        self.cfg = model_cfg
        self.out_dim = self.cfg.MODEL.PA_MLP.wave_out_dim
        self.in_ch = self.cfg.INIT.IO_RIR_out_ch
        self.conv1 = layer.DownBlock(self.in_ch, 16, 64, 4, True)
        self.conv2 = layer.DownBlock(16, 64, 32, 4, True)
        self.conv3 = layer.DownBlock(64, 128, 16, 4, True)
        self.conv4 = layer.DownBlock(128, self.out_dim, 8, 4, True)
        self.conv5 = layer.DownBlock(self.out_dim, self.out_dim, 4, 4, True)
        self.conv6 = layer.Conv1Block(self.out_dim, self.out_dim, 2,
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
        self.out_dim = self.cfg.MODEL.PA_MLP.stft_out_dim
        self.n_Mel = self.cfg.MODEL.PA_MLP.n_Mel
        self.spec_input = torchaudio.transforms.MelSpectrogram(
            sample_rate=48000, n_fft=1024, hop_length=128,
            norm='slaney', n_mels=self.n_Mel)
        self.conv1 = layer.upBlock(self.n_Mel * self.in_ch, self.n_Mel * 4,7,
                                   2,True, 'prelu', 4)
        self.conv2 = layer.upBlock(self.n_Mel * 4, self.n_Mel * 4,7,
                                   2,True,'prelu', None)
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






