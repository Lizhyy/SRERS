import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.parallel

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import train.Model.model_layer as layer
import train.Model.RIR_decoder.Dec_layer as Dec_layer
import train.Model.RIR_decoder.ER_Dec_block as EDB
import train.Model.RIR_decoder.PA_Dec_block as PDB
import train.Model.RIR_decoder.LR_Dec_block as LRDB
from CConfig import config

cfg = config()


class RIR_decoder(nn.Module):
    """Fixed SRE-MHA-LoR decoder; outputs early waveform, parameters, and late envelope."""

    def __init__(self, model_cfg=cfg):
        super(RIR_decoder, self).__init__()
        self.cfg = model_cfg
        self.decoder = SRERS_Decoder(model_cfg=self.cfg)

    def forward(self, mesh_embed, real_ERs, ls_src_pos):
        ER, para, LR = self.decoder(mesh_embed, real_ERs, ls_src_pos)
        return (ER, para, LR), None


class SRERS_Decoder(nn.Module):
    def __init__(self, model_cfg=cfg):
        super(SRERS_Decoder, self).__init__()
        self.cfg = model_cfg
        self.mesh_2_ebd = MESH_seq_ebd_MHA(model_cfg=self.cfg)
        self.mesh_2_ebd1 = MESH_seq_ebd_MHA(model_cfg=self.cfg)
        self.mesh_2_ebd2 = MESH_seq_ebd_MHA(model_cfg=self.cfg)
        self.ER_dec = EDB.SRE_ER_Decoder(model_cfg=self.cfg)
        self.ER_dec1 = PDB.SRE_PA_Decoder(model_cfg=self.cfg)
        self.ER_dec2 = LRDB.SRE_LR_Decoder(model_cfg=self.cfg)

    def forward(self, mesh_embed, ER_batch_ori, lst_src_pos_ori):
        mesh_query = self.mesh_2_ebd(mesh_embed, ER_batch_ori.clone(), lst_src_pos_ori.clone())
        SRIR_er = self.ER_dec(mesh_query, ER_batch_ori.clone(), lst_src_pos_ori.clone())

        mesh_query1 = self.mesh_2_ebd1(mesh_embed, ER_batch_ori.clone(), lst_src_pos_ori.clone())
        SRIR_para = self.ER_dec1(mesh_query1, ER_batch_ori.clone(), lst_src_pos_ori.clone())

        mesh_query2 = self.mesh_2_ebd2(mesh_embed, ER_batch_ori.clone(), lst_src_pos_ori.clone())
        SRIR_lr = self.ER_dec2(mesh_query2, ER_batch_ori.clone(), lst_src_pos_ori.clone())
        return SRIR_er, SRIR_para, SRIR_lr


class MESH_seq_ebd_MHA(nn.Module):
    def __init__(self, model_cfg=cfg):
        super(MESH_seq_ebd_MHA, self).__init__()
        self.cfg = model_cfg
        self.query_seq_length = self.cfg.MODEL.TF_CNN.query_seq_length
        self.pos_dim = self.cfg.MODEL.TF_CNN.TF_pos_dim
        self.GCN_layers_channel_out = self.cfg.MODEL.SRE_GCN.GCN_layers_channel_out
        self.rir_fc_in_dim = self.GCN_layers_channel_out * self.query_seq_length
        self.pos_enc_1 = layer.POS_ENC(feature_dim_in=self.pos_dim * 9,
                                        feature_dim_out=self.rir_fc_in_dim)
        self.MHA_decoder = Dec_layer.Transformer_MHA_Decoder(
            d_model=self.GCN_layers_channel_out,
            nhead=4,
            num_layers=1,
            dim_feedforward=self.GCN_layers_channel_out,
            dropout=0.1,
        )
        self.ebd2rir_fc = layer.fcBlock(in_planes=self.GCN_layers_channel_out + self.pos_dim,
                                        out_planes=self.GCN_layers_channel_out,
                                        bias=False,
                                        BN=True,
                                        nonlin_ly='prelu')

    def forward(self, mesh_embed, ER_batch, lst_src_pos):
        batch_size, _ = lst_src_pos.size()
        pos_ebd_1 = self.pos_enc_1(lst_src_pos)
        pos_ebd_1 = pos_ebd_1.view(batch_size, self.query_seq_length, self.GCN_layers_channel_out)
        mesh_query = self.MHA_decoder(pos_ebd_1, mesh_embed)
        mesh_query = mesh_query.view(batch_size, -1)
        full_embed = torch.cat((mesh_query, lst_src_pos), dim=1)
        mesh_query = self.ebd2rir_fc(full_embed)
        return mesh_query
