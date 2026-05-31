import sys
from pathlib import Path

import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from CConfig import config
import train.Model.MESH_encoder.MESH_block as MEB

cfg = config()


class MESH_encoder(nn.Module):
    """Fixed SRE-GCN-MGL-MLP mesh encoder for the GSLR release."""

    def __init__(self, mesh_cfg=cfg):
        super(MESH_encoder, self).__init__()
        self.cfg = mesh_cfg
        self.mesh_net = SRE_mesh_ebd_model(model_cfg=self.cfg)

    def forward(self, data):
        x, edge_index, batch, embeddings = data.x, data.edge_index, data.batch, data.embeddings
        RIR = data.RIR
        ER = data.ER.float()
        para = data.para
        reverb_ev = data.reverb_ev
        x = self.mesh_net(x, edge_index, batch)
        return x, RIR, ER, para, reverb_ev, embeddings, data.RIR_path


class SRE_mesh_ebd_model(nn.Module):
    def __init__(self, model_cfg=cfg):
        super(SRE_mesh_ebd_model, self).__init__()
        self.cfg = model_cfg
        self.GCN_Block = MEB.GCN_Block(model_cfg=self.cfg)
        self.MESH_EBD = MEB.Mesh_ebd_MLP_block(model_cfg=self.cfg)

    def forward(self, x, edge_index, batch):
        x, edge_index, batch = self.GCN_Block(x, edge_index, batch)
        x = self.MESH_EBD(x, edge_index, batch)
        return x
