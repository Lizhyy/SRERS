import os
import pickle
import random as rd
import sys
from pathlib import Path

import numpy as np
import torch
import torch.utils.data as data
import torch_geometric as pyg

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import train.para_dataset.para_remaker as enc


class SRERS_RIR_para_Dataset(data.Dataset):
    """GSLR-only dataset for the minimal SRE-GCN-MGL-MLP / SRE-MHA-LoR release."""

    def __init__(self, data_cfg, dataset_name, zero_graph=False):
        if data_cfg.INIT.Dataset != 'GSLR':
            raise ValueError('Minimal publish version only supports the GSLR dataset.')
        if dataset_name not in ('train', 'valid', 'debug_train', 'debug_valid'):
            raise ValueError("dataset_name must be one of: 'train', 'valid', 'debug_train', 'debug_valid'.")
        self.cfg = data_cfg
        self.dataset_name = dataset_name
        self.zero_graph = zero_graph
        self.embeddings = self.load_embedding()

    def get_RIR(self, RIR_para_full_path):
        RIR_para_dic = enc.SRIR_para_dic_loader(
            RIR_para_full_path,
            tag_check=self.cfg.INIT.data_tag,
            data_cfg=self.cfg,
            re_write=False,
        )
        SRIR_ER_norm, para, reverb_norm_ev, LoR_cut = enc.Para_load_from_dic(
            self.cfg,
            RIR_para_dic,
            RIR_para_full_path,
        )
        ori_rir = np.zeros(2, dtype=np.float32)
        return SRIR_ER_norm, para, reverb_norm_ev, LoR_cut, ori_rir

    def get_graph(self, full_graph_path, zero_graph=False):
        if zero_graph:
            return pyg.data.Data(
                x=torch.zeros((3, 32), dtype=torch.float32),
                edge_index=torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
            )

        with open(full_graph_path, 'rb') as f:
            graph_dic = pickle.load(f)

        if 'face_graph' not in graph_dic or 'face_edge_tr' not in graph_dic:
            raise KeyError('GSLR face graph requires keys: face_graph and face_edge_tr.')

        x = torch.from_numpy(graph_dic['face_graph'].astype('float32'))
        edge_index = torch.from_numpy(graph_dic['face_edge_tr']).long()
        return pyg.data.Data(x=x, edge_index=edge_index)

    def load_embedding(self):
        if self.dataset_name == 'train':
            ebd_path = self.cfg.DATA.train_embed_path
        elif self.dataset_name == 'valid':
            ebd_path = self.cfg.DATA.valid_embed_path
        elif self.dataset_name == 'debug_train':
            ebd_path = self.cfg.DATA.debug_train_embed_path
        elif self.dataset_name == 'debug_valid':
            ebd_path = self.cfg.DATA.debug_valid_embed_path
        else:
            raise ValueError(self.dataset_name)

        with open(ebd_path, 'rb') as f:
            embeddings = pickle.load(f)
        rd.shuffle(embeddings)
        return sorted(embeddings)

    def __getitem__(self, index):
        graph_path, para_full_path, source_location, receiver_location = self.embeddings[index]
        source_receiver = source_location + receiver_location
        embedding = np.array(source_receiver, dtype=np.float32)
        graph = self.get_graph(graph_path, zero_graph=self.zero_graph)
        SRIR_ER_norm, para, reverb_norm_ev, LoR_cut, _ = self.get_RIR(para_full_path)

        graph.RIR = np.array([SRIR_ER_norm], dtype=np.float32)
        graph.para = para.astype(np.float32)
        graph.reverb_ev = np.array([reverb_norm_ev], dtype=np.float32)
        graph.embeddings = embedding.astype(np.float32)
        graph.ER = np.array([LoR_cut], dtype=np.float32)
        graph.RIR_path = para_full_path
        return graph

    def __len__(self):
        return len(self.embeddings)
