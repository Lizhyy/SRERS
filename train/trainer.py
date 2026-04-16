import os
import random as rd
import re

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torch_geometric.nn
from tqdm import tqdm

from Config import config, save_cfg
import train.Loss.Loss_dic as ls_dic
import train.Loss.loss_function as lsf
import train.miscc.Utils as uls
from train.miscc.Utils import mkdir_p, save_model
from train.miscc.mesh_extend import (
    RIR_batch_switch,
    ebd_switch,
    ev_batch_switch,
    x_batch_switch,
)

cfg = config()
print(str(cfg.TRAIN.GPU_id)[1:-1])

if cfg.INIT.random_seed_set:
    rd.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)


class SRERSTrainer(object):
    """Training loop for the scene encoder + SRIR parameter decoder."""

    def __init__(self, train_cfg, output_dir):
        self.cfg = train_cfg
        self.model_dir = os.path.join(output_dir, 'Model')
        self.rir_dir = os.path.join(output_dir, 'RIR')
        mkdir_p(self.model_dir)
        mkdir_p(self.rir_dir)
        save_cfg(self.cfg, output_dir)

        self.max_epoch = self.cfg.TRAIN.max_epoch
        self.lr_decay_rate = self.cfg.TRAIN.LR_decay_rate
        self.gpus = self.cfg.TRAIN.GPU_id
        self.num_gpus = len(self.gpus)
        self.batch_size = self.cfg.TRAIN.batch_size * self.num_gpus

        torch.cuda.set_device(self.gpus[0])
        cudnn.benchmark = True

        rd.seed(0)
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)


    def _prepare_batch(self, data_list, allow_random_rotation=True):
        prepared_batch = []
        for data in data_list:
            switch_type = rd.randint(0, 5) if allow_random_rotation else 0

            data['RIR'] = torch.from_numpy(RIR_batch_switch(data['RIR'], switch_type)).float()
            data['ER'] = torch.from_numpy(RIR_batch_switch(data['ER'], switch_type)).float()
            data['reverb_ev'] = torch.from_numpy(ev_batch_switch(data['reverb_ev'], switch_type)).float()
            sample_id = data['RIR_path']
            data['RIR_path'] = torch.from_numpy(
                np.array([[int(x) for x in re.findall(r'\d+', sample_id)]])
            )
            data['para'] = torch.from_numpy(np.array([data['para']])).float()
            data['x'] = x_batch_switch(data['x'], switch_type).float()
            data['embeddings'] = torch.from_numpy(
                np.array([ebd_switch(data['embeddings'], switch_type)])
            ).float()
            prepared_batch.append(data)

        return prepared_batch


    def _run_one_epoch(self, scene_encoder, srir_decoder, data_loader, optimizer_mesh=None, optimizer_decoder=None, max_steps=None):
        loss_meter = {}
        is_train = optimizer_mesh is not None and optimizer_decoder is not None

        for step_idx, data_list in tqdm(enumerate(data_loader), total=max_steps):
            batch = self._prepare_batch(data_list, allow_random_rotation=is_train)

            if cfg.INIT.random_seed_set and step_idx == 0:
                print(batch[-1]['RIR_path'])

            (
                scene_feature_sequence,
                target_early_residual,
                input_lor_waveform,
                target_aux_params,
                target_late_reverb_envelope,
                source_listener_coords,
                _,
            ) = scene_encoder(batch)

            decoder_inputs = (scene_feature_sequence, input_lor_waveform, source_listener_coords)
            decoder_outputs, _ = nn.parallel.data_parallel(srir_decoder, decoder_inputs, self.gpus)
            pred_early_residual, pred_aux_params, pred_late_reverb_envelope = decoder_outputs

            loss_dict = lsf.compute_total_srir_loss(
                target_early_residual=target_early_residual,
                target_aux_params=target_aux_params,
                target_late_reverb_envelope=target_late_reverb_envelope,
                pred_early_residual=pred_early_residual,
                pred_aux_params=pred_aux_params,
                pred_late_reverb_envelope=pred_late_reverb_envelope,
            )

            if is_train:
                scene_encoder.zero_grad()
                srir_decoder.zero_grad()
                total_loss = loss_dict['tensor']
                total_loss.float().backward()
                optimizer_mesh.step()
                optimizer_decoder.step()

            loss_meter = ls_dic.Loss_Dic_maker(loss_meter, loss_dict)

            if max_steps is not None and step_idx > 0 and step_idx % max_steps == 0:
                break

        return ls_dic.Loss_Dic_mean(loss_meter)


    def train(self, train_loader, test_loader):
        if cfg.INIT.random_seed_set:
            rd.seed(0)
            torch.manual_seed(0)
            torch.cuda.manual_seed_all(0)

        scene_encoder, srir_decoder, discriminator = uls.load_model_train(self.cfg)
        scene_encoder = torch_geometric.nn.DataParallel(scene_encoder, device_ids=self.gpus)
        scene_encoder.to(device=f'cuda:{self.gpus[0]}')
        srir_decoder.to(device=f'cuda:{self.gpus[0]}')

        optimizer_mesh = optim.RMSprop(scene_encoder.parameters(), lr=self.cfg.TRAIN.MESH_LR)
        optimizer_decoder = optim.RMSprop(srir_decoder.parameters(), lr=self.cfg.TRAIN.GENERATOR_LR)

        mesh_lr = self.cfg.TRAIN.MESH_LR
        decoder_lr = self.cfg.TRAIN.GENERATOR_LR
        lr_decay_step = self.cfg.TRAIN.LR_decay_epoch
        min_loss = self.cfg.TRAIN.min_loss

        print('Min_Loss:', min_loss)

        for epoch in range(self.max_epoch):
            if epoch % lr_decay_step == 0 and epoch > 0:
                decoder_lr *= self.lr_decay_rate
                for param_group in optimizer_decoder.param_groups:
                    param_group['lr'] = decoder_lr

                mesh_lr *= self.lr_decay_rate
                for param_group in optimizer_mesh.param_groups:
                    param_group['lr'] = mesh_lr

            if epoch < self.cfg.TRAIN.start_epoch:
                continue

            train_loss_dict = self._run_one_epoch(
                scene_encoder=scene_encoder,
                srir_decoder=srir_decoder,
                data_loader=train_loader,
                optimizer_mesh=optimizer_mesh,
                optimizer_decoder=optimizer_decoder,
                max_steps=self.cfg.TRAIN.loss_show,
            )

            print(f'Train  Epoch:{epoch}  model:{self.cfg.DATA.out_folder}')
            ls_dic.Loss_Dic_mean_show(train_loss_dict)
            uls.save_newest_model(scene_encoder, srir_decoder, discriminator, self.model_dir, epoch)

            max_valid_epoch = round(cfg.TRAIN.loss_show / 5)

            with torch.no_grad():
                valid_loss_dict = self._run_one_epoch(
                    scene_encoder=scene_encoder,
                    srir_decoder=srir_decoder,
                    data_loader=test_loader,
                    optimizer_mesh=None,
                    optimizer_decoder=None,
                    max_steps=max_valid_epoch,
                )

                if valid_loss_dict['total_LOSS'] < min_loss:
                    min_loss = valid_loss_dict['total_LOSS']
                    save_model(scene_encoder, srir_decoder, discriminator, 'best', self.model_dir)
                    print(f'!!!!!! epoch:{epoch}  BEST model !!!!!!')

                print(
                    f'Valid  Epoch:{epoch}  model:{self.cfg.DATA.out_folder}  \n'
                    f'MIN_Loss:{np.round(min_loss, 2)}'
                )
                ls_dic.Loss_Dic_mean_show(valid_loss_dict, self.model_dir)
                print('  ')


# Backward-compatible alias
GANTrainer = SRERSTrainer
