import json
import os
import pickle
from pathlib import Path
import sys

from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from CConfig import config

cfg = config()


def _dataset_paths(dataset_name):
    if dataset_name == 'train':
        return cfg.DATA.train_dataset_path, cfg.DATA.train_embed_path
    if dataset_name == 'valid':
        return cfg.DATA.valid_dataset_path, cfg.DATA.valid_embed_path
    raise ValueError("Minimal publish version only supports dataset_name='train' or 'valid'.")


def embed_generator(dataset_name, re_make=True):
    """Generate the GSLR embedding index for train/validation only."""
    dataset_path, ebd_path = _dataset_paths(dataset_name)
    if (not re_make) and os.path.exists(ebd_path):
        return None

    ebd_list = []
    for room_name in tqdm(os.listdir(dataset_path)):
        try:
            room_path = os.path.join(dataset_path, room_name)
            graph_path = os.path.join(room_path, 'house_sp_sim_graph.pickle')
            rir_folder_path = os.path.join(room_path, 'para_full')
            json_path = os.path.join(room_path, 'sim_config.json')
            with open(json_path, encoding='utf-8') as f:
                loc_json = json.load(f)

            for listener_idx, receiver in enumerate(loc_json['receivers']):
                for source_idx, source_info in enumerate(loc_json['sources']):
                    rir_name = f'SRIR_S{source_idx}_L{str(listener_idx).zfill(4)}.pickle'
                    rir_path = os.path.join(rir_folder_path, rir_name)
                    if os.path.exists(rir_path):
                        ebd_list.append([graph_path, rir_path, source_info['xyz'], receiver['xyz']])
        except Exception:
            continue

    os.makedirs(os.path.dirname(ebd_path), exist_ok=True)
    with open(ebd_path, 'wb') as f:
        pickle.dump(ebd_list, f, protocol=2)
    return ebd_list


if __name__ == '__main__':
    embed_generator('train', re_make=True)
    embed_generator('valid', re_make=True)
