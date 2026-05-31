import os
import json


# --------------------------------------------------------
# Minimal publish configuration
# --------------------------------------------------------
# The released version is intentionally fixed to the main model:
# SRE-GCN-MGL-MLP_SRE-MHA-LoR_10%-face-reverb-4-od2_GSLR

Debug = False
Train_GPU_ID = [0]
Train_Batch_size = 32
loss_show = 3000
num_workers = 32
RIR_type = 'reverb-mch-wav'

start_epoch = 0
min_loss = 1e10

with_LoR = True


class config():
    def __init__(self):
        self.USE_GPU = True
        self.INIT = INIT()
        self.DATA = DATA(self.INIT.mesh_encoder, self.INIT.RIR_decoder, self.INIT.IO_info)
        self.TRAIN = TRAIN()
        self.TEST = TEST()
        self.MODEL = MODEL(self.INIT.IO_mesh_in_vertex_ebd,
                           self.DATA.RIR_input.RIR_size,
                           self.DATA.RIR_input.RIR_wave_size)


class INIT():
    def __init__(self):
        # Fixed main mesh encoder: SRE-GCN-MGL-MLP
        self.mesh_ebd_model = 'SRE'
        self.mesh_ebd_gcn_type = 'GCN'
        self.mesh_ebd_mesh_sim = 'MGL'
        self.mesh_ebd_mesh_enc = 'MLP'

        # Fixed main SRIR decoder: SRE-MHA-LoR
        self.RIR_dec_model = 'SRE'
        self.RIR_pos_query = 'MHA'
        self.LoR_ebd_block = 'LoR'

        # Fixed I/O setup: 10%-face-reverb-4-od2
        self.IO_mesh_in_vertex_num = '10%'
        self.IO_mesh_in_vertex_ebd = 'face'
        self.IO_RIR_ER_type = 'reverb'
        self.IO_RIR_out_ch = 4
        self.IO_LoR_od = 'od2'

        # Fixed dataset: GSLR only
        self.Dataset = 'GSLR'
        self.data_tag = 5

        self.HS_5090 = True
        self.random_seed_set = False

        self.mesh_encoder = (self.mesh_ebd_model + '-' +
                             self.mesh_ebd_gcn_type + '-' +
                             self.mesh_ebd_mesh_sim + '-' +
                             self.mesh_ebd_mesh_enc)
        self.RIR_decoder = (self.RIR_dec_model + '-' +
                            self.RIR_pos_query + '-' +
                            self.LoR_ebd_block)
        self.IO_info = (self.IO_mesh_in_vertex_num + '-' +
                        self.IO_mesh_in_vertex_ebd + '-' +
                        self.IO_RIR_ER_type + '-' +
                        str(self.IO_RIR_out_ch) + '-' +
                        self.IO_LoR_od)

        self.ER_input = False


class RIR_input():
    def __init__(self):
        self.RIR_wave_size = 4096
        self.RIR_std_size = 256
        self.RIR_size = self.RIR_wave_size + self.RIR_std_size
        self.additional_para = True
        self.RIR_type = RIR_type
        self.mch_gain = 8
        self.mono_gain = 4


class DATA():
    def __init__(self, mesh_encoder, RIR_decoder, IO_info):
        self.RIR_input = RIR_input()
        self.dataset_path = ''

        self.train_dataset_path = os.path.join(self.dataset_path, 'dataset_train')
        self.valid_dataset_path = os.path.join(self.dataset_path, 'dataset_valid')
        self.test_dataset_path = os.path.join(self.dataset_path, 'dataset_test')

        self.debug_train_dataset_path = os.path.join(self.dataset_path, 'dataset_small/valid')
        self.debug_valid_dataset_path = os.path.join(self.dataset_path, 'dataset_small/valid')
        self.debug_test_dataset_path = os.path.join(self.dataset_path, 'dataset_small/test')

        self.train_embed_path = os.path.join(self.dataset_path, 'ebd/ebd_train.pickle')
        self.valid_embed_path = os.path.join(self.dataset_path, 'ebd/ebd_valid.pickle')
        self.test_embed_path = os.path.join(self.dataset_path, 'ebd/ebd_test.pickle')

        self.debug_train_embed_path = os.path.join(self.dataset_path, 'ebd/ebd_dbg_train.pickle')
        self.debug_valid_embed_path = os.path.join(self.dataset_path, 'ebd/ebd_dbg_valid.pickle')
        self.debug_test_embed_path = os.path.join(self.dataset_path, 'ebd/ebd_dbg_test.pickle')

        self.DEBUG = Debug
        self.debug = '_debug' if self.DEBUG else ''

        self.out_path = os.path.join(
            '',
            f'{mesh_encoder}_{RIR_decoder}_{IO_info}_GSLR{self.debug}',
        )
        self.out_folder = os.path.basename(self.out_path)


class TRAIN():
    def __init__(self):
        self.start_epoch = start_epoch
        self.min_loss = min_loss
        self.batch_size = Train_Batch_size
        self.max_epoch = 150
        self.LR_decay_epoch = 1
        self.LR_decay_rate = 0.88
        self.loss_show = loss_show
        self.GPU_id = Train_GPU_ID
        self.num_workers = num_workers

        self.GENERATOR_LR = 0.00004
        self.MESH_LR = 0.00004

        if self.start_epoch == 'best':
            self.MESH_NET = 'Mesh_Enc_epoch_best.pth'
            self.NET_G = 'RIR_G_epoch_best.pth'
        elif self.start_epoch == 0:
            self.MESH_NET = ''
            self.NET_G = ''
        else:
            self.MESH_NET = 'Mesh_Enc_epoch_{}.pth'.format(str(self.start_epoch - 1).zfill(4))
            self.NET_G = 'RIR_G_epoch_{}.pth'.format(str(self.start_epoch - 1).zfill(4))


class TEST():
    def __init__(self):
        self.GPU_id = Train_GPU_ID
        self.batch_size = Train_Batch_size
        self.num_workers = num_workers
        self.with_LoR = with_LoR
        self.Folder_name = 'RIR'
        self.MESH_NET = 'Mesh_Enc_epoch_best.pth'
        self.NET_G = 'RIR_G_epoch_best.pth'


class MODEL():
    def __init__(self, vertex, RIR_size, RIR_wave_size):
        self.SRE_GCN = SRE_GCN(vertex)
        self.ER_ENC = ER_ENC()
        self.TF_CNN = TF_CNN(self.SRE_GCN.mesh_seq_length, RIR_wave_size)
        self.PA_MLP = PA_MLP()
        self.LR_CNN = LR_CNN()


class SRE_GCN():
    def __init__(self, vertex):
        if vertex == 'face':
            self.mesh_feature_dim = 32
        else:
            raise ValueError("Minimal publish version only supports IO_mesh_in_vertex_ebd='face'.")
        self.GCN_layers_channel_in = [256, 512, 1024]
        self.GCN_layers_channel_out = 1024
        self.GCN_layers_rate = 0.5
        self.mesh_seq_length = 256


class ER_ENC():
    def __init__(self):
        self.echo_segment_size = 64
        self.echo_histogram_bins = 16
        self.echo_stride = 16
        self.echo_frame_num = (4096 - self.echo_segment_size) // self.echo_stride + 1
        self.echo_channel = 128
        self.ER_frame_num = (4096 - self.echo_segment_size) // 32 + 1


class TF_CNN():
    def __init__(self, mesh_seq_length, RIR_wave_size):
        self.query_seq_length = 1
        self.mesh_seq_length = mesh_seq_length
        self.RIR_feature_len = RIR_wave_size // 128
        self.NF_MLP_dim = 1024
        self.RIR_feature_channel = 512
        self.condition_dim = 10
        self.TF_pos_dim = 6
        self.LoR_pos_ebd = 256
        self.Pos_sin_enc_repeat = 8
        self.CNN_k_size = [33, 37, 41, 45, 49, 53, 57, 61, 65]


class PA_MLP():
    def __init__(self):
        self.n_Mel = 64
        self.mlp_dim = 1024
        self.wave_out_dim = 256
        self.stft_out_dim = 256
        self.en_dim = 256


class LR_CNN():
    def __init__(self):
        self.n_Mel = 64
        self.mlp_dim = 4096
        self.mlp_out = 5120
        self.cnn_in_ch = 4
        self.wave_out_dim = 512
        self.stft_out_dim = 512
        self.en_dim = 256


def to_dict(obj):
    if isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    if isinstance(obj, (list, tuple, set)):
        return [to_dict(i) for i in obj]
    if hasattr(obj, '__dict__'):
        return {k: to_dict(v) for k, v in vars(obj).items()}
    return str(obj)


def save_cfg(cfg, path):
    cfg_dict = to_dict(cfg)
    path_out = os.path.join(path, 'cfg.json')
    with open(path_out, 'w', encoding='utf-8') as f:
        json.dump(cfg_dict, f, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    cfg = config()
    print(cfg.DATA.out_folder)
