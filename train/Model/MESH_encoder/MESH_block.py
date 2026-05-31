from pathlib import Path
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from CConfig import config
import train.Model.MESH_encoder.MESH_layer as layer

cfg = config()


class Mesh_ebd_MLP_block(nn.Module):
    """Keep pooled GCN node embeddings as the fixed mesh token sequence."""

    def __init__(self, model_cfg=cfg):
        super().__init__()
        self.cfg = model_cfg
        self.mesh_seq_length = self.cfg.MODEL.SRE_GCN.mesh_seq_length
        self.GCN_layers_channel_out = self.cfg.MODEL.SRE_GCN.GCN_layers_channel_out

    def forward(self, x, edge_index, batch):
        return x.view(torch.max(batch) + 1, self.mesh_seq_length, self.GCN_layers_channel_out)


class GCN_Block(nn.Module):
    """Fixed SRE-GCN-MGL graph encoder block for the face graph input."""

    def __init__(self, model_cfg=cfg):
        super().__init__()
        self.cfg = model_cfg
        self.input_feature_dim = self.cfg.MODEL.SRE_GCN.mesh_feature_dim
        self.GCN_layers_channel_in = self.cfg.MODEL.SRE_GCN.GCN_layers_channel_in
        self.GCN_layers_channel_out = self.cfg.MODEL.SRE_GCN.GCN_layers_channel_out
        self.GCN_layers_rate = self.cfg.MODEL.SRE_GCN.GCN_layers_rate
        self.mesh_seq_length = self.cfg.MODEL.SRE_GCN.mesh_seq_length
        self.GCN_type = self.cfg.INIT.mesh_ebd_gcn_type

        self.GCN_layers_1 = layer.Mesh_En_GCN_layer(
            feature_dim_in=self.input_feature_dim,
            feature_dim_out=self.GCN_layers_channel_in[0],
            ratio=self.GCN_layers_rate,
            GCN_layer=self.GCN_type,
            Gpool_layer='TopK',
            NL='prelu',
        )
        self.GCN_layers_2 = layer.Mesh_En_GCN_layer(
            feature_dim_in=self.GCN_layers_channel_in[0],
            feature_dim_out=self.GCN_layers_channel_in[1],
            ratio=self.GCN_layers_rate,
            GCN_layer=self.GCN_type,
            Gpool_layer='TopK',
            NL='prelu',
        )
        self.GCN_layers_3 = layer.Mesh_En_GCN_layer(
            feature_dim_in=self.GCN_layers_channel_in[1],
            feature_dim_out=self.GCN_layers_channel_in[2],
            ratio=self.mesh_seq_length * 3,
            GCN_layer=self.GCN_type,
            Gpool_layer='TopK',
            NL='prelu',
        )
        self.GCN_layers_4 = layer.Mesh_En_GCN_layer(
            feature_dim_in=self.GCN_layers_channel_in[2],
            feature_dim_out=self.GCN_layers_channel_out,
            ratio=self.mesh_seq_length,
            GCN_layer=self.GCN_type,
            Gpool_layer='TopK',
            NL='prelu',
        )

    def forward(self, x, edge_index, batch):
        x, edge_index, batch = self.GCN_layers_1(x, edge_index, batch)
        x, edge_index, batch = self.GCN_layers_2(x, edge_index, batch)
        x, edge_index, batch = self.GCN_layers_3(x, edge_index, batch)
        x, edge_index, batch = self.GCN_layers_4(x, edge_index, batch)
        x = F.dropout(x, p=0.5, training=self.training)
        return x, edge_index, batch
