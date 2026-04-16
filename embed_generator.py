"""Build the sample index used by the SRERS dataset loader."""

import argparse
import json
import os
import pickle
from typing import List

from tqdm import tqdm

from Config import config


def build_sample_index(dataset_name: str, re_make: bool = True) -> List[list]:
    """Create or reload the list of `(graph_path, para_path, source, listener)` tuples."""
    cfg = config()

    if dataset_name == 'train':
        dataset_path = cfg.DATA.train_dataset_path
        index_path = cfg.DATA.train_embed_path
    elif dataset_name == 'test':
        dataset_path = cfg.DATA.test_dataset_path
        index_path = cfg.DATA.test_embed_path
    else:
        raise ValueError(f'Unsupported dataset split: {dataset_name}')

    if not re_make and os.path.exists(index_path):
        with open(index_path, 'rb') as f:
            return pickle.load(f)

    room_name_list = os.listdir(dataset_path)
    sample_index = []

    for room_name in tqdm(room_name_list, desc=f'Indexing {dataset_name}'):
        room_path = os.path.join(dataset_path, room_name)
        graph_path = os.path.join(room_path, 'house_sp_sim_graph.pickle')
        rir_folder_path = os.path.join(room_path, 'para_full')
        json_path = os.path.join(room_path, 'sim_config.json')

        try:
            with open(json_path, 'r', encoding='utf-8') as json_file:
                location_config = json.load(json_file)

            num_listener = len(location_config['receivers'])
            num_sources = len(location_config['sources'])

            for listener_idx in range(num_listener):
                for source_idx in range(num_sources):
                    source_location = location_config['sources'][source_idx]['xyz']
                    listener_location = location_config['receivers'][listener_idx]['xyz']
                    rir_name = f'SRIR_S{source_idx}_L{str(listener_idx).zfill(4)}.pickle'
                    rir_path = os.path.join(rir_folder_path, rir_name)

                    if os.path.exists(rir_path):
                        sample_index.append([graph_path, rir_path, source_location, listener_location])
        except Exception:
            # Skip malformed rooms while preserving the behavior of the original script.
            continue

    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    with open(index_path, 'wb') as f:
        pickle.dump(sample_index, f, protocol=2)

    return sample_index


embed_generator = build_sample_index


def build_argparser():
    parser = argparse.ArgumentParser(description='Build SRERS dataset index files.')
    parser.add_argument('--data-root', type=str, default=None, help='Override SRERS_DATA_ROOT.')
    parser.add_argument('--reuse-existing', action='store_true', help='Reuse existing index files if found.')
    return parser


def main():
    args = build_argparser().parse_args()
    if args.data_root:
        os.environ['SRERS_DATA_ROOT'] = args.data_root

    train_index = build_sample_index('train', re_make=not args.reuse_existing)
    test_index = build_sample_index('test', re_make=not args.reuse_existing)
    print(f'train samples: {len(train_index)}')
    print(f'test samples: {len(test_index)}')


if __name__ == '__main__':
    main()
