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
import train.Model.RIR_decoder.Dec_layer as layer
import train.Model.RIR_decoder.ER_LoR_Enc_block as LoR_B
from CConfig import config
cfg = config()
import numpy as np



class SRE_ER_Decoder(nn.Module):
    def __init__(self, model_cfg=cfg):
        super(SRE_ER_Decoder, self).__init__()
        self.cfg = model_cfg
        self.upsample = ER_LoR_decoder(model_cfg=self.cfg)

    def forward(self, mesh_query, ER_batch, lst_src_pos):
        RIR_wav = self.upsample(mesh_query, ER_batch, lst_src_pos)
        return RIR_wav




class ER_LoR_decoder(nn.Module):
    def __init__(self, model_cfg=cfg):
        super(ER_LoR_decoder, self).__init__()
        self.cfg = model_cfg
        self.RIR_feature_channel = self.cfg.MODEL.TF_CNN.RIR_feature_channel
        self.out_channel = self.cfg.INIT.IO_RIR_out_ch
        self.RIR_feature_len = cfg.MODEL.TF_CNN.RIR_feature_len
        self.query_seq_length = self.cfg.MODEL.TF_CNN.query_seq_length
        self.k_size = self.cfg.MODEL.TF_CNN.CNN_k_size
        self.GCN_layers_channel_out = self.cfg.MODEL.SRE_GCN.GCN_layers_channel_out
        self.rir_fc_in_dim = self.GCN_layers_channel_out * self.query_seq_length

        self.RIR_fc = layer.fcBlock(self.rir_fc_in_dim,
                                    self.RIR_feature_channel * self.RIR_feature_len,
                                    bias=False)

        self.upsample1 = layer.upBlock(
            self.RIR_feature_channel, self.RIR_feature_channel,
            self.k_size[0], 2, True, 'prelu')
        # self.upsample2 = layer.upBlock(
        #     self.RIR_feature_channel, self.RIR_feature_channel,
        #     self.k_size[1], 2, True, 'prelu')
        self.upsample3 = layer.upBlock(
            self.RIR_feature_channel, self.RIR_feature_channel // 2,
            self.k_size[2], 2, True, 'prelu')

        self.ER_enc = LoR_B.LoR_encoder_old(self.cfg)
        self.ebd_add = layer.Conv1Block(
            self.RIR_feature_channel, self.RIR_feature_channel,
            1, True)

        self.upsample4 = layer.upBlock(
            self.RIR_feature_channel, self.RIR_feature_channel,
            self.k_size[3], 2, True, 'prelu')
        self.upsample5 = layer.upBlock(
            self.RIR_feature_channel, self.RIR_feature_channel // 2,
            self.k_size[4], 2, True, 'prelu')
        self.upsample6 = layer.upBlock(
            self.RIR_feature_channel // 2, self.RIR_feature_channel // 2,
            self.k_size[5], 2, True, 'prelu')
        self.upsample7 = layer.upBlock(
            self.RIR_feature_channel // 2, self.RIR_feature_channel // 4,
            self.k_size[6], 2, True, 'prelu')
        self.upsample8 = layer.upBlock(
            self.RIR_feature_channel // 4, self.RIR_feature_channel // 4,
            self.k_size[7], 2, True, 'prelu')
        self.RIR_out_layer = layer.Conv1Block(
            self.RIR_feature_channel // 4, self.out_channel,
            self.k_size[8], False, 'none')


    def forward(self, mesh_query, ER_batch, lst_src_pos):
        ER_ebd = self.RIR_fc(mesh_query)
        ER_ebd = ER_ebd.view(-1, self.RIR_feature_channel, self.RIR_feature_len)

        ER_ebd = self.upsample1(ER_ebd)
        # ER_ebd = self.upsample2(ER_ebd)
        ER_ebd = self.upsample3(ER_ebd)
        LoR_ebd = self.ER_enc(ER_batch)
        ER_wav = torch.cat([ER_ebd, LoR_ebd], dim=1)
        ER_wav = self.ebd_add(ER_wav)
        ER_wav = self.upsample4(ER_wav)
        ER_wav = self.upsample5(ER_wav)
        ER_wav = self.upsample6(ER_wav)
        ER_wav = self.upsample7(ER_wav)
        ER_wav = self.upsample8(ER_wav)
        ER_wav = self.RIR_out_layer(ER_wav)
        return ER_wav

