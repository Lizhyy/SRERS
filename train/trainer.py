import os
import random as rd
import re
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch_geometric.nn
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import train.Loss.loss_function as lsf
import train.Loss.Loss_dic as ls_dic
import train.miscc.Utils as uls
from CConfig import save_cfg
from train.miscc.Utils import mkdir_p, save_model
from train.miscc.mesh_extend import ebd_switch, ev_batch_switch, RIR_batch_switch, x_batch_switch


def _path_to_index_tensor(rir_path: str) -> torch.Tensor:
    numbers = [int(x) for x in re.findall(r'\d+', str(rir_path))]
    if len(numbers) == 0:
        numbers = [0]
    return torch.from_numpy(np.array([numbers], dtype=np.int64))


class GANTrainer(object):
    def __init__(self, train_cfg, output_dir):
        self.cfg = train_cfg
        self.model_dir = os.path.join(output_dir, 'Model')
        self.RIR_dir = os.path.join(output_dir, 'RIR')
        mkdir_p(output_dir)
        mkdir_p(self.model_dir)
        mkdir_p(self.RIR_dir)
        save_cfg(self.cfg, output_dir)

        self.max_epoch = self.cfg.TRAIN.max_epoch
        self.lr_de_rate = self.cfg.TRAIN.LR_decay_rate
        self.gpus = self.cfg.TRAIN.GPU_id
        self.num_gpus = len(self.gpus)
        self.batch_size = self.cfg.TRAIN.batch_size * self.num_gpus

        torch.cuda.set_device(self.gpus[0])
        cudnn.benchmark = True
        rd.seed(0)
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)

    def _prepare_data_list(self, data_list, augment: bool):
        prepared = []
        for data in data_list:
            switch_type = rd.randint(0, 5) if augment else 0
            data['RIR'] = torch.from_numpy(RIR_batch_switch(data['RIR'], switch_type)).float()
            data['ER'] = torch.from_numpy(RIR_batch_switch(data['ER'], switch_type)).float()
            data['reverb_ev'] = torch.from_numpy(ev_batch_switch(data['reverb_ev'], switch_type)).float()
            data['RIR_path'] = _path_to_index_tensor(data['RIR_path'])
            data['para'] = torch.from_numpy(np.array([data['para']])).float()
            data['x'] = x_batch_switch(data['x'], switch_type).float()
            data['embeddings'] = torch.from_numpy(np.array([ebd_switch(data['embeddings'], switch_type)])).float()
            prepared.append(data)
        return prepared

    def _forward_loss(self, Mesh_Enc, RIR_G, data_list_batch):
        mesh_ebd, real_ER, real_LoR, real_para, real_LR, ls_src_pos, _ = Mesh_Enc(data_list_batch)
        fake_ER, _ = nn.parallel.data_parallel(RIR_G, (mesh_ebd, real_LoR, ls_src_pos), self.gpus)
        fake_ER, fake_para, fake_LR = fake_ER[0], fake_ER[1], fake_ER[2]
        return lsf.Loss_RIR_all(
            real_ER=real_ER,
            real_para=real_para,
            real_LR=real_LR,
            fake_ER=fake_ER,
            fake_para=fake_para,
            fake_LR=fake_LR,
        )

    def train(self, train_loader, valid_loader):
        if self.cfg.INIT.random_seed_set:
            rd.seed(0)
            torch.manual_seed(0)
            torch.cuda.manual_seed_all(0)

        Mesh_Enc, RIR_G, _ = uls.load_model_train(self.cfg)
        Mesh_Enc = torch_geometric.nn.DataParallel(Mesh_Enc, device_ids=self.gpus)
        Mesh_Enc.to(device='cuda:{}'.format(str(self.gpus[0])))
        RIR_G.to(device='cuda:{}'.format(str(self.gpus[0])))

        optimizer_Mesh = optim.RMSprop(Mesh_Enc.parameters(), lr=self.cfg.TRAIN.MESH_LR)
        optimizer_RIRG = optim.RMSprop(RIR_G.parameters(), lr=self.cfg.TRAIN.GENERATOR_LR)

        generator_lr = self.cfg.TRAIN.GENERATOR_LR
        mesh_lr = self.cfg.TRAIN.MESH_LR
        lr_decay_step = self.cfg.TRAIN.LR_decay_epoch
        min_loss = self.cfg.TRAIN.min_loss
        print('Min_Loss:', min_loss)

        for epoch in range(self.max_epoch):
            if epoch % lr_decay_step == 0 and epoch > 0:
                generator_lr *= self.lr_de_rate
                mesh_lr *= self.lr_de_rate
                for param_group in optimizer_RIRG.param_groups:
                    param_group['lr'] = generator_lr
                for param_group in optimizer_Mesh.param_groups:
                    param_group['lr'] = mesh_lr

            if epoch < self.cfg.TRAIN.start_epoch:
                continue

            Mesh_Enc.train()
            RIR_G.train()
            train_batch_loss_dic = {}
            train_total = min(len(train_loader), self.cfg.TRAIN.loss_show + 1)
            for i, data_list in tqdm(enumerate(train_loader), total=train_total):
                data_list_batch = self._prepare_data_list(data_list, augment=True)
                Mesh_Enc.zero_grad()
                RIR_G.zero_grad()
                loss_dic = self._forward_loss(Mesh_Enc, RIR_G, data_list_batch)
                loss_dic['tensor'].float().backward()
                optimizer_Mesh.step()
                optimizer_RIRG.step()
                train_batch_loss_dic = ls_dic.Loss_Dic_maker(train_batch_loss_dic, loss_dic)
                if (i > 0) and (i % self.cfg.TRAIN.loss_show == 0):
                    break

            print('Train  Epoch:{}  model:{}'.format(epoch, self.cfg.DATA.out_folder))
            train_loss_dic = ls_dic.Loss_Dic_mean(train_batch_loss_dic)
            ls_dic.Loss_Dic_mean_show(train_loss_dic)
            uls.save_newest_model(Mesh_Enc, RIR_G, None, self.model_dir, epoch)

            Mesh_Enc.eval()
            RIR_G.eval()
            valid_batch_loss_dic = {}
            valid_total = len(valid_loader)
            with torch.no_grad():
                for data_list in tqdm(valid_loader, total=valid_total):
                    data_list_batch = self._prepare_data_list(data_list, augment=False)
                    loss_dic = self._forward_loss(Mesh_Enc, RIR_G, data_list_batch)
                    valid_batch_loss_dic = ls_dic.Loss_Dic_maker(valid_batch_loss_dic, loss_dic)

            valid_loss_dic = ls_dic.Loss_Dic_mean(valid_batch_loss_dic)
            if valid_loss_dic['total_LOSS'] < min_loss:
                min_loss = valid_loss_dic['total_LOSS']
                save_model(Mesh_Enc, RIR_G, None, 'best', self.model_dir)
                print('!!!!!! epoch:{}  BEST model !!!!!!'.format(epoch))

            print('Valid  Epoch:{}  model:{}  \nMIN_Loss:{}'.format(
                epoch,
                self.cfg.DATA.out_folder,
                np.round(min_loss, 2)))
            ls_dic.Loss_Dic_mean_show(valid_loss_dic, self.model_dir)
            print('  ')
