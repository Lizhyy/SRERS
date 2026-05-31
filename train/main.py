import os
import random as rd
import resource
import sys
from pathlib import Path

import torch
from torch_geometric.loader.data_list_loader import DataListLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from CConfig import config
from train.para_dataset.datasets import SRERS_RIR_para_Dataset
from trainer import GANTrainer


def set_seed(seed: int = 0):
    rd.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def raise_file_limit(target_soft_limit: int = 65535):
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (min(target_soft_limit, hard), hard))


if __name__ == '__main__':
    cfg = config()
    print('conda env:', os.environ.get('CONDA_DEFAULT_ENV'))
    raise_file_limit()
    set_seed(0)

    num_gpu = len(cfg.TRAIN.GPU_id)
    train_dataset = SRERS_RIR_para_Dataset(data_cfg=cfg, dataset_name='train')
    valid_dataset = SRERS_RIR_para_Dataset(data_cfg=cfg, dataset_name='valid')

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
        shuffle=False,
        num_workers=int(cfg.TRAIN.num_workers),
        drop_last=False,
    )

    print(cfg.DATA.out_path)
    trainer = GANTrainer(cfg, cfg.DATA.out_path)
    trainer.train(train_loader, valid_loader)
