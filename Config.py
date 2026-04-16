import json
import os
from typing import Any, List


def _parse_gpu_ids_from_env(default_gpu_ids: list[int]) -> list[int]:
    """Parse `SRERS_GPU_IDS` like `0,1` into a list of ints."""
    gpu_env = os.environ.get('SRERS_GPU_IDS', '').strip()
    if not gpu_env:
        return default_gpu_ids
    try:
        return [int(x.strip()) for x in gpu_env.split(',') if x.strip()]
    except ValueError:
        return default_gpu_ids


def _env_flag(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {'1', 'true', 'yes', 'y', 'on'}

def _first_existing_path(*candidates: str) -> str:
    """Return the first non-empty candidate that already exists on disk.

    If none of the candidates exists, return the last non-empty candidate so the
    caller still gets a deterministic writable/expected path.
    """
    non_empty = [c for c in candidates if c]
    for candidate in non_empty:
        if os.path.exists(candidate):
            return candidate
    return non_empty[-1] if non_empty else ''


loss_show = 3000
GAN_G_additional = 2
num_workers = 32
RIR_type = 'reverb-mch-wav'
add_tag = ''
with_LoR = True

start_epoch = 0
min_loss = 1e10

Test_Batch_size = 32
out_folder_name = 'RIR'


class config:
    """Project-wide configuration.

    This class keeps the original structure used by the training code,
    while adding clearer comments and a few safer defaults for public use.
    """

    def __init__(self):
        self.USE_GPU = _env_flag('SRERS_USE_GPU', True)
        self.INIT = INIT()
        self.DATA = DATA()
        self.TRAIN = TRAIN()
        self.TEST = TEST()
        self.MODEL = MODEL(
            self.INIT.IO_mesh_in_vertex_ebd,
            self.DATA.RIR_input.RIR_size,
            self.DATA.RIR_input.RIR_wave_size,
        )


class INIT:
    """High-level experiment switches.

    The names below are kept for backward compatibility with older checkpoints.
    """

    def __init__(self):
        # Scene encoder
        self.mesh_ebd_gcn_type = 'GCN'    # GCN, GAT
        self.mesh_ebd_mesh_enc = 'MLP'    # TF, MLP

        # SRIR decoder
        self.RIR_pos_query = 'TF'         # TF, MHA, MLP
        self.LoR_ebd_block = 'LoR'        # LoR, wo, LoRnew, LoRnew1

        # I/O representation
        self.IO_mesh_in_vertex_num = '10%'    # 2k, 10%
        self.IO_mesh_in_vertex_ebd = 'face'   # pos, pos1k, posfull, face
        self.IO_RIR_ER_type = 'RIR'           # RIR, reverb
        self.IO_RIR_out_ch = 4
        self.IO_LoR_od = 'od2'

        self.random_seed_set = False
        self.data_tag = 5
        self.ER_input = False


class RIR_input:
    def __init__(self):
        self.RIR_wave_size = 4096
        self.RIR_std_size = 256
        self.RIR_size = self.RIR_wave_size + self.RIR_std_size
        self.additional_para = True
        self.RIR_type = RIR_type
        self.mch_gain = 8
        self.mono_gain = 4


class DATA:
    """Dataset and output paths.

    Override these with environment variables when moving the project:
    - SRERS_DATA_ROOT
    - SRERS_OUTPUT_ROOT
    """

    def __init__(self):
        self.RIR_input = RIR_input()

        default_data_root = '/mnt/disk_work1/lzy/SRERS/GSRE'
        portable_data_root = './data'
        default_output_root = './outputs'

        self.dataset_path = _first_existing_path(
            os.environ.get('SRERS_DATA_ROOT', ''),
            default_data_root,
            portable_data_root,
        )
        self.train_dataset_path = os.path.join(self.dataset_path, 'dataset_train')
        self.test_dataset_path = os.path.join(self.dataset_path, 'dataset_test')

        self.train_embed_path = os.path.join(self.dataset_path, 'ebd/ebd_train.pickle')
        self.test_embed_path = os.path.join(self.dataset_path, 'ebd/ebd_test.pickle')

        self.out_path = os.environ.get('SRERS_OUTPUT_ROOT', default_output_root)
        self.out_folder = os.path.basename(self.out_path.rstrip('/')) or 'SRERS'


class TRAIN:
    def __init__(self):
        self.start_epoch = start_epoch
        self.min_loss = min_loss
        self.batch_size = 16
        self.max_epoch = 150
        self.LR_decay_epoch = 1
        self.LR_decay_rate = 0.85
        self.loss_show = loss_show
        self.GPU_id = _parse_gpu_ids_from_env([0, 1])
        self.num_workers = num_workers

        self.DISCRIMINATOR_LR = 0.00012
        self.GENERATOR_LR = 0.00004
        self.MESH_LR = 0.00004

        if self.start_epoch == 'best':
            self.MESH_NET = 'Mesh_Enc_epoch_best.pth'
            self.NET_G = 'RIR_G_epoch_best.pth'
            self.NET_D = 'RIR_D_epoch_best.pth'
        elif self.start_epoch == 0:
            self.MESH_NET = ''
            self.NET_G = ''
            self.NET_D = ''
        else:
            self.MESH_NET = f'Mesh_Enc_epoch_{str(self.start_epoch - 1).zfill(4)}.pth'
            self.NET_G = f'RIR_G_epoch_{str(self.start_epoch - 1).zfill(4)}.pth'
            self.NET_D = f'RIR_D_epoch_{str(self.start_epoch - 1).zfill(4)}.pth'

        self.GAN_G_additional = GAN_G_additional


class TEST:
    def __init__(self):
        self.batch_size = Test_Batch_size
        self.GPU_id = _parse_gpu_ids_from_env([0, 1])
        self.num_workers = 32

        self.MESH_NET = 'Mesh_Enc_epoch_best.pth'
        self.NET_G = 'RIR_G_epoch_best.pth'
        self.NET_D = 'RIR_D_epoch_best.pth'

        self.with_LoR = with_LoR
        self.Folder_name = out_folder_name


class MODEL:
    def __init__(self, vertex, RIR_size, RIR_wave_size):
        self.SRE_GCN = SRE_GCN(vertex)

        self.ER_ENC = ER_ENC()
        self.TF_CNN = TF_CNN(self.SRE_GCN.mesh_seq_length, RIR_wave_size)
        self.PA_MLP = PA_MLP()
        self.LR_CNN = LR_CNN()


class SRE_GCN:
    def __init__(self, vertex):
        self.mesh_feature_dim = 32
        self.GCN_layers_channel_in = [256, 512, 1024]
        self.GCN_layers_channel_out = 1024
        self.GCN_layers_rate = 0.5
        self.mesh_seq_length = 256


class ER_ENC:
    def __init__(self):
        self.echo_segment_size = 64
        self.echo_histogram_bins = 16
        self.echo_stride = 16
        self.echo_frame_num = (4096 - self.echo_segment_size) // self.echo_stride + 1
        self.echo_channel = 128
        self.ER_frame_num = (4096 - self.echo_segment_size) // 32 + 1


class TF_CNN:
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


class PA_MLP:
    def __init__(self):
        self.n_Mel = 64
        self.mlp_dim = 1024
        self.wave_out_dim = 256
        self.stft_out_dim = 256
        self.en_dim = 256


class LR_CNN:
    def __init__(self):
        self.n_Mel = 64
        self.mlp_dim = 4096
        self.mlp_out = 5120
        self.cnn_in_ch = 4
        self.wave_out_dim = 512
        self.stft_out_dim = 512
        self.en_dim = 256


def to_dict(obj: Any):
    """Recursively convert a config object into JSON-serializable data."""
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
