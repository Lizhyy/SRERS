"""Training entry point for SRERS."""

import argparse
import os

from Config import config
from torch_geometric.loader.data_list_loader import DataListLoader
from train.para_dataset.datasets import SRERSParameterDataset
from train.trainer import SRERSTrainer


def build_argparser():
    parser = argparse.ArgumentParser(description='Train SRERS.')
    parser.add_argument('--data-root', type=str, default=None, help='Override SRERS_DATA_ROOT.')
    parser.add_argument('--output-root', type=str, default=None, help='Override SRERS_OUTPUT_ROOT.')
    return parser


def main():
    args = build_argparser().parse_args()
    if args.data_root:
        os.environ['SRERS_DATA_ROOT'] = args.data_root
    if args.output_root:
        os.environ['SRERS_OUTPUT_ROOT'] = args.output_root

    cfg = config()
    env_name = os.environ.get('CONDA_DEFAULT_ENV', '<unknown>')
    print('conda env:', env_name)

    output_dir = cfg.DATA.out_path
    num_gpu = len(cfg.TRAIN.GPU_id)

    train_dataset = SRERSParameterDataset(data_cfg=cfg, dataset_name='train')
    valid_dataset = SRERSParameterDataset(data_cfg=cfg, dataset_name='test')

    train_loader = DataListLoader(
        train_dataset,
        batch_size=cfg.TRAIN.batch_size * num_gpu,
        shuffle=True,
        num_workers=int(cfg.TRAIN.num_workers),
        drop_last=True,
    )
    valid_loader = DataListLoader(
        valid_dataset,
        batch_size=cfg.TRAIN.batch_size * num_gpu,
        shuffle=True,
        num_workers=int(cfg.TRAIN.num_workers),
        drop_last=True,
    )

    print('output_dir:', output_dir)
    trainer = SRERSTrainer(cfg, output_dir)
    trainer.train(train_loader, valid_loader)


if __name__ == '__main__':
    main()
