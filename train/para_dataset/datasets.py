from Config import config
cfg = config()

import pickle
import random as rd

import numpy as np
import torch
import torch.utils.data as data
import torch_geometric as pyg

import train.para_dataset.para_remaker as enc


if cfg.INIT.random_seed_set:
    rd.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)


class SRERSParameterDataset(data.Dataset):
    """Dataset that returns one scene graph and one SRIR parameter sample.

    Preferred semantic field names added to each `pyg.data.Data` sample:
        - target_early_residual
        - input_lor_waveform
        - target_aux_params
        - target_late_reverb_envelope
        - source_listener_coords
        - sample_id

    Legacy field names are still attached for backward compatibility:
        - RIR, ER, para, reverb_ev, embeddings, RIR_path
    """

    def __init__(self, data_cfg, dataset_name, zero_graph=False):
        self.cfg = data_cfg
        self.dataset_name = dataset_name
        self.sample_index = self.load_embedding()
        self.zero_graph = zero_graph

    def get_rir_targets(self, rir_parameter_path):
        rewrite_cached_file = False
        rir_parameter_dict = enc.SRIR_para_dic_loader(
            rir_parameter_path,
            tag_check=self.cfg.INIT.data_tag,
            data_cfg=self.cfg,
            re_write=rewrite_cached_file,
        )
        early_residual, aux_params, late_reverb_envelope, lor_waveform = enc.Para_load_from_dic(
            self.cfg,
            rir_parameter_dict,
            rir_parameter_path,
        )
        original_rir = np.zeros(2)
        return early_residual, aux_params, late_reverb_envelope, lor_waveform, original_rir

    def get_graph(self, full_graph_path, zero_graph=False):
        if zero_graph:
            return pyg.data.Data()

        if self.cfg.INIT.IO_mesh_in_vertex_num == '2k':
            full_graph_path = full_graph_path.replace(
                'house_sp_sim_graph.pickle',
                'house_sp_sim_graph_2000.pickle',
            )

        with open(full_graph_path, 'rb') as f:
            graph_dict = pickle.load(f)

        vertex_feature_mode = self.cfg.INIT.IO_mesh_in_vertex_ebd
        if vertex_feature_mode == 'pos':
            node_features = torch.from_numpy(graph_dict['pos'][:, :3].astype('float32'))
        elif vertex_feature_mode == 'posfull':
            node_features = torch.from_numpy(graph_dict['pos'].astype('float32'))
        elif vertex_feature_mode == 'pos1k':
            node_features = torch.from_numpy(graph_dict['pos'][:, [0, 1, 2, 6, 14]].astype('float32'))
        elif vertex_feature_mode == 'face':
            node_features = torch.from_numpy(graph_dict['face_graph'].astype('float32'))
        else:
            raise ValueError(f'Unsupported vertex embedding mode: {vertex_feature_mode}')

        if vertex_feature_mode == 'face':
            edge_index = torch.from_numpy(graph_dict['face_edge_tr'])
        else:
            edge_index = torch.from_numpy(graph_dict['edge_index'])

        return pyg.data.Data(x=node_features, edge_index=edge_index)

    def load_embedding(self):
        if self.dataset_name == 'train':
            index_path = self.cfg.DATA.train_embed_path
        elif self.dataset_name == 'test':
            index_path = self.cfg.DATA.test_embed_path
        else:
            raise ValueError(f'Unsupported dataset split: {self.dataset_name}')

        with open(index_path, 'rb') as f:
            sample_index = pickle.load(f)

        if self.dataset_name == 'test':
            rd.seed(0)
            torch.manual_seed(0)
            torch.cuda.manual_seed_all(0)
            sample_index = rd.sample(sample_index, min(8192 * 2, len(sample_index)))
        else:
            rd.shuffle(sample_index)

        return sample_index

    def __getitem__(self, index):
        graph_path, para_full_path, source_location, receiver_location = self.sample_index[index]
        source_listener_coords = np.array(source_location + receiver_location).astype('float32')
        graph = self.get_graph(graph_path, zero_graph=self.zero_graph)

        target_early_residual, target_aux_params, target_late_reverb_envelope, input_lor_waveform, original_rir = (
            self.get_rir_targets(para_full_path)
        )

        # Preferred semantic names
        graph.target_early_residual = np.array([target_early_residual]).astype('float32')
        graph.target_aux_params = target_aux_params.astype('float32')
        graph.target_late_reverb_envelope = np.array([target_late_reverb_envelope]).astype('float32')
        graph.input_lor_waveform = np.array([input_lor_waveform]).astype('float32')
        graph.source_listener_coords = source_listener_coords.astype('float32')
        graph.sample_id = original_rir if cfg.INIT.Dataset == 'GWA' else para_full_path

        # Backward-compatible names used by the original code
        graph.RIR = graph.target_early_residual
        graph.para = graph.target_aux_params
        graph.reverb_ev = graph.target_late_reverb_envelope
        graph.ER = graph.input_lor_waveform
        graph.embeddings = graph.source_listener_coords
        graph.RIR_path = graph.sample_id

        return graph

    def __len__(self):
        return len(self.sample_index)


# Backward-compatible alias
SRERS_RIR_para_Dataset = SRERSParameterDataset
