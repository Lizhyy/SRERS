"""Evaluation / export entry point for SRERS."""

import argparse
import os
import random
import re
import resource
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch_geometric
from thop import profile
from torch_geometric.loader.data_list_loader import DataListLoader
from tqdm import tqdm

import train.Loss.Loss_dic as ls_dic
import train.miscc.Utils as uls
import train.miscc.inference_RIR_out as rout
from Config import config
from train.miscc.mesh_extend import (
    RIR_batch_switch,
    ebd_switch,
    ev_batch_switch,
    x_batch_switch,
)
from train.para_dataset.datasets import SRERSParameterDataset


class SRERSEvaluator(object):
    def __init__(self, test_cfg, output_dir):
        self.cfg = test_cfg
        self.model_dir = os.path.join(output_dir, 'Model')
        self.rir_dir = os.path.join(output_dir, self.cfg.TEST.Folder_name)

        self.gpus = self.cfg.TEST.GPU_id
        self.num_gpus = len(self.gpus)
        self.batch_size = self.cfg.TEST.batch_size * self.num_gpus
        torch.cuda.set_device(self.gpus[0])
        cudnn.benchmark = True

    def _prepare_batch(self, data_list):
        prepared_batch = []
        for data in data_list:
            switch_type = 0
            data['RIR'] = torch.from_numpy(RIR_batch_switch(data['RIR'], switch_type)).float()
            data['ER'] = torch.from_numpy(RIR_batch_switch(data['ER'], switch_type)).float()
            data['reverb_ev'] = torch.from_numpy(ev_batch_switch(data['reverb_ev'], switch_type)).float()
            sample_id = data['RIR_path']
            data['RIR_path'] = torch.from_numpy(np.array([[int(x) for x in re.findall(r'\d+', sample_id)]]))

            data['para'] = torch.from_numpy(np.array([data['para']])).float()
            data['x'] = x_batch_switch(data['x'], switch_type).float()
            data['embeddings'] = torch.from_numpy(
                np.array([ebd_switch(data['embeddings'], switch_type)])
            ).float()
            prepared_batch.append(data)
        return prepared_batch

    def test(self, test_loader):
        scene_encoder, srir_decoder = uls.load_model_test(self.cfg)
        scene_encoder = torch_geometric.nn.DataParallel(scene_encoder, device_ids=self.gpus)
        scene_encoder.to(device=f'cuda:{self.gpus[0]}')
        srir_decoder.to(device=f'cuda:{self.gpus[0]}')

        with torch.no_grad():
            test_batch_loss_dict = {}
            mesh_time = 0.0
            decoder_time = 0.0

            for batch_idx, data_list in tqdm(enumerate(test_loader), total=len(test_loader)):
                batch = self._prepare_batch(data_list)

                if batch_idx < 1:
                    flops, params = profile(scene_encoder, inputs=(batch,))
                    print('Scene encoder FLOPs:', flops / 1e6)
                    print('Scene encoder Params:', params / 1e6)

                start_t = time.time()
                (
                    scene_feature_sequence,
                    target_early_residual,
                    input_lor_waveform,
                    target_aux_params,
                    target_late_reverb_envelope,
                    source_listener_coords,
                    sample_id,
                ) = scene_encoder(batch)
                mesh_time += time.time() - start_t

                if batch_idx < 1:
                    flops, params = profile(
                        srir_decoder,
                        inputs=(scene_feature_sequence, input_lor_waveform, source_listener_coords),
                    )
                    print('SRIR decoder FLOPs:', round(flops / 1e6, 4))
                    print('SRIR decoder Params:', round(params / 1e6, 4))

                start_t = time.time()
                decoder_outputs, _ = nn.parallel.data_parallel(
                    srir_decoder,
                    (scene_feature_sequence, input_lor_waveform, source_listener_coords),
                    self.gpus,
                )
                decoder_time += time.time() - start_t

                pred_early_residual, pred_aux_params, pred_late_reverb_envelope = decoder_outputs

                rout.SRERS_full_RIR_decoder(
                    target_early_residual,
                    input_lor_waveform,
                    target_aux_params,
                    target_late_reverb_envelope,
                    pred_early_residual,
                    pred_aux_params,
                    pred_late_reverb_envelope,
                    sample_id,
                    self.rir_dir,
                    batch_idx,
                )
                rout.SRERS_separate_RIR_decoder(
                    [target_early_residual, target_aux_params, target_late_reverb_envelope],
                    pred_early_residual,
                    sample_id,
                    input_lor_waveform,
                )

            print('\nTest finished')
            print('Scene encoder time:', mesh_time / (batch_idx + 1) * 1000, 'ms')
            print('SRIR decoder time:', decoder_time / (batch_idx + 1) * 1000, 'ms')

            test_loss_dict = ls_dic.Loss_Dic_mean(test_batch_loss_dict)
            ls_dic.Loss_Dic_mean_show(test_loss_dict, self.model_dir)


GANTester = SRERSEvaluator


def build_argparser():
    parser = argparse.ArgumentParser(description='Evaluate or export SRERS predictions.')
    parser.add_argument('--data-root', type=str, default=None, help='Override SRERS_DATA_ROOT.')
    parser.add_argument('--output-root', type=str, default=None, help='Override SRERS_OUTPUT_ROOT.')
    parser.add_argument(
        '--nofile-limit',
        type=int,
        default=65535,
        help='Temporary RLIMIT_NOFILE soft limit.',
    )
    return parser


def main():
    args = build_argparser().parse_args()
    if args.data_root:
        os.environ['SRERS_DATA_ROOT'] = args.data_root
    if args.output_root:
        os.environ['SRERS_OUTPUT_ROOT'] = args.output_root

    cfg = config()
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    print(f'Original RLIMIT_NOFILE: soft={soft}, hard={hard}')
    resource.setrlimit(resource.RLIMIT_NOFILE, (args.nofile_limit, hard))
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    print(f'Updated RLIMIT_NOFILE: soft={soft}, hard={hard}')

    random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    output_dir = cfg.DATA.out_path
    num_gpu = len(cfg.TEST.GPU_id)

    test_dataset = SRERSParameterDataset(cfg, 'test')
    test_loader = DataListLoader(
        test_dataset,
        batch_size=cfg.TEST.batch_size * num_gpu,
        shuffle=False,
        num_workers=int(cfg.TRAIN.num_workers),
        drop_last=True,
    )

    print('output_dir:', output_dir)
    evaluator = SRERSEvaluator(cfg, output_dir)
    evaluator.test(test_loader)


if __name__ == '__main__':
    main()
