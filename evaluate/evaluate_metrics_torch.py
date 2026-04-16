import os
import torch
from tqdm import tqdm
from Config import config
import train.miscc.Data_IO as dio
import numpy as np
import pickle
from train.para_dataset.SRIR_encoder import sig_len_reshape
import pandas as pd
cfg = config()

import torch.utils.data as data
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import torch.nn as nn
from torchaudio.transforms import MelSpectrogram


class Dataset_evaluate(data.Dataset):
    def __init__(self, demo_folder_path, RIR_len=48000 * 3):
        self.demo_folder_path = demo_folder_path
        self.RIR_path_list = self.get_RIR_path_list()
        self.RIR_len = RIR_len


    def get_RIR_path_list(self):
        room_list = os.listdir(self.demo_folder_path)
        RIR_path_list = []
        print('get rir path: ')
        for room_name in tqdm(room_list):
            room_path = os.path.join(self.demo_folder_path, room_name)
            for src in range(20):
                for lsr in range(50):
                    real_wav_name = 'SRIR_S{}_L{}_real.wav'.format(str(src), str(lsr).zfill(4))
                    real_wav_path = os.path.join(room_path, real_wav_name)
                    if not os.path.exists(real_wav_path):
                        if not 'BOX' in room_path:
                            continue

                    fake_wav_name = 'SRIR_S{}_L{}_infe.wav'.format(str(src), str(lsr).zfill(4))
                    fake_wav_path = os.path.join(room_path, fake_wav_name)
                    if not os.path.exists(fake_wav_path):
                        continue

                    ori_wav_name = 'SRIR_S{}_L{}_ori.wav'.format(str(src), str(lsr).zfill(4))
                    ori_wav_path = os.path.join(room_path, ori_wav_name)
                    if not os.path.exists(ori_wav_path):
                        continue

                    if not 'BOX' in room_path:
                        RIR_path_list.append((ori_wav_path, real_wav_path, fake_wav_path))
                    else:
                        RIR_path_list.append((ori_wav_path, ori_wav_path, fake_wav_path))
        return RIR_path_list



    def __getitem__(self, index):
        ori_wav_path, real_wav_path, fake_wav_path = self.RIR_path_list[index]
        ori_wav = np.array(sig_len_reshape(dio.load(ori_wav_path), self.RIR_len))
        real_wav = np.zeros_like(ori_wav)
        fake_wav = np.array(sig_len_reshape(dio.load(fake_wav_path), self.RIR_len))
        if np.ndim(ori_wav) == 1:
            ori_wav = np.array([ori_wav])
        if np.ndim(real_wav) == 1:
            real_wav = np.array([real_wav])
        if np.ndim(fake_wav) == 1:
            fake_wav = np.array([fake_wav])

        ori_wav = torch.from_numpy(ori_wav)
        real_wav = torch.from_numpy(real_wav)
        fake_wav = torch.from_numpy(fake_wav)
        return ori_wav, real_wav, fake_wav


    def __len__(self):
        return len(self.RIR_path_list)




class Masked_Loss(nn.Module):
    def __init__(self, L_type):
        super(Masked_Loss, self).__init__()
        self.L_type = L_type

    def forward(self, input, target, mask):
        if self.L_type == 1:
            loss = torch.abs(input - target)
        else:
            loss = (input - target) ** self.L_type
        loss = loss * mask
        return loss.sum() / mask.sum().clamp(min=1)  # 避免除以 0





class Evaluator(object):
    def __init__(self, GPU_id, batch_size, RIR_len=48000 * 3):
        self.RIR_len = RIR_len
        self.gpus = GPU_id
        self.num_gpus = len(self.gpus)
        self.batch_size = batch_size * self.num_gpus
        torch.cuda.set_device(self.gpus[0])
        cudnn.benchmark = True

        self.Mel_spec_time = MelSpectrogram(sample_rate=48000, n_fft=64, hop_length=16,
                                            n_mels=64 // 16, norm='slaney',
                                            power=1).to(device='cuda').double()
        self.Mel_spec_norm = MelSpectrogram(sample_rate=48000, n_fft=256, hop_length=64,
                                            n_mels=256 // 16, norm='slaney',
                                            power=1).to(device='cuda').double()
        self.Mel_spec_freq = MelSpectrogram(sample_rate=48000, n_fft=2048, hop_length=512,
                                            n_mels=2048 // 16, norm='slaney',
                                            power=1).to(device='cuda').double()

    def evaluate(self, data_loader):
        with torch.no_grad():
            L1_Mask_loss_compute = Masked_Loss(1)
            L2_Mask_loss_compute = Masked_Loss(2)
            L1_loss_compute = nn.L1Loss()
            L2_loss_compute = nn.MSELoss()

            wav_L1_err = torch.tensor(0.0).to(device='cuda')
            wav_L2_err = torch.tensor(0.0).to(device='cuda')
            T60_err = torch.tensor(0.0).to(device='cuda')
            En_err = torch.tensor(0.0).to(device='cuda')
            DRR_err = torch.tensor(0.0).to(device='cuda')
            Mel_time_err = torch.tensor(0.0).to(device='cuda')
            Mel_freq_err = torch.tensor(0.0).to(device='cuda')
            Mel_norm_err = torch.tensor(0.0).to(device='cuda')

            print('computing: ')
            for j, wav in tqdm(enumerate(data_loader), total=len(data_loader)):
                ori_wav, _, infe_wav = wav
                ori_wav = ori_wav.to(device='cuda')
                infe_wav = infe_wav.to(device='cuda')
                ori_mask, ori_en_T60 = self.mask_maker_3D_with_T60_en(ori_wav)
                infe_mask, infe_en_T60 = self.mask_maker_3D_with_T60_en(infe_wav)
                mask = (ori_mask.float() + infe_mask.float()).bool()
                input = (ori_wav, infe_wav, mask)
                wav_L1_err += nn.parallel.data_parallel(L1_Mask_loss_compute, input, self.gpus)
                wav_L2_err += nn.parallel.data_parallel(L2_Mask_loss_compute, input, self.gpus)

                ori_T60 = self.T60_en_2_T60(ori_en_T60)
                infe_T60 = self.T60_en_2_T60(infe_en_T60)
                input = (ori_T60, infe_T60)
                T60_err += nn.parallel.data_parallel(L1_loss_compute, input, self.gpus)

                ori_en = self.sig_energy(ori_wav)
                infe_en = self.sig_energy(infe_wav)
                input = (self.to_dB(ori_en), self.to_dB(infe_en))
                En_err += nn.parallel.data_parallel(L1_loss_compute, input, self.gpus)

                ori_imp_en = self.sig_energy(ori_wav[:, :, :240])
                ori_DRR = self.to_dB(ori_imp_en / (ori_en - ori_imp_en))
                ori_DRR = self.replace_invalid(ori_DRR, 160)

                infe_imp_en = self.sig_energy(infe_wav[:, :, :240])
                infe_DRR = self.to_dB(infe_imp_en / (infe_en - infe_imp_en))
                infe_DRR = self.replace_invalid(infe_DRR, 160)

                input = (ori_DRR, infe_DRR)
                DRR_err += nn.parallel.data_parallel(L1_loss_compute, input, self.gpus)

                ori_Mel_time, ori_Mel_time_mask = self.RIR_MelSpectrogram_mask(ori_wav, 'time')
                infe_Mel_time, infe_Mel_time_mask = self.RIR_MelSpectrogram_mask(infe_wav, 'time')
                Mel_time_mask = ori_Mel_time_mask + infe_Mel_time_mask
                input = (ori_Mel_time, infe_Mel_time, Mel_time_mask)
                Mel_time_err += nn.parallel.data_parallel(L1_Mask_loss_compute, input, self.gpus)

                ori_Mel_norm, ori_Mel_norm_mask = self.RIR_MelSpectrogram_mask(ori_wav, 'norm')
                infe_Mel_norm, infe_Mel_norm_mask = self.RIR_MelSpectrogram_mask(infe_wav, 'norm')
                Mel_norm_mask = ori_Mel_norm_mask + infe_Mel_norm_mask
                input = (ori_Mel_norm, infe_Mel_norm, Mel_norm_mask)
                Mel_norm_err += nn.parallel.data_parallel(L1_Mask_loss_compute, input, self.gpus)

                ori_Mel_freq, ori_Mel_freq_mask = self.RIR_MelSpectrogram_mask(ori_wav, 'freq')
                infe_Mel_freq, infe_Mel_freq_mask = self.RIR_MelSpectrogram_mask(infe_wav, 'freq')
                Mel_freq_mask = ori_Mel_freq_mask + infe_Mel_freq_mask
                input = (ori_Mel_freq, infe_Mel_freq, Mel_freq_mask)
                Mel_freq_err += nn.parallel.data_parallel(L1_Mask_loss_compute, input, self.gpus)

            wav_L1_err = (wav_L1_err / len(data_loader)).cpu().item()
            print('L1_err: ', wav_L1_err)
            wav_L2_err = (wav_L2_err / len(data_loader)).cpu().item()
            print('L2_err: ', wav_L2_err)
            T60_err = (T60_err / len(data_loader)).cpu().item()
            print('T60_err: ', T60_err)
            En_err = (En_err / len(data_loader)).cpu().item()
            print('En_err: ', En_err)
            DRR_err = (DRR_err / len(data_loader)).cpu().item()
            print('DRR_err: ', DRR_err)
            Mel_freq_err = (Mel_freq_err / len(data_loader)).cpu().item()
            print('Mel_freq_err: ', Mel_freq_err)
            Mel_norm_err = (Mel_norm_err / len(data_loader)).cpu().item()
            print('Mel_norm_err: ', Mel_norm_err)
            Mel_time_err = (Mel_time_err / len(data_loader)).cpu().item()
            print('Mel_time_err: ', Mel_time_err)
            return {'L1_err': wav_L1_err, 'L2_err': wav_L2_err, 'T60_err': T60_err,
                    'En_err': En_err, 'DRR_err': DRR_err, 'Mel_freq_err': Mel_freq_err,
                    'Mel_norm_err': Mel_norm_err, 'Mel_time_err': Mel_time_err}


    def mask_maker_3D_with_T60_en(self, batch_tensor_in):
        batch_tensor = batch_tensor_in.clone().to(device='cuda')
        # 沿着 channel 维度求绝对值和，得到逐时间点的能量
        frame_energy = torch.flip(
            torch.cumsum(
                torch.flip(batch_tensor ** 2, dims=(2,)), dim=2)
            , dims=(2,)
        ) # [B, T]
        # 找到最后一个非零位置
        frame_energy2 = frame_energy.clone().to(device='cuda')
        nonzero_mask = (frame_energy2 != 0)  # [B, T]
        mask = nonzero_mask.float().sum(dim=1).bool().unsqueeze(1).repeat(1, 4, 1)
        # mask_np = mask.numpy()
        # batch_np = batch_tensor.numpy()
        return mask, frame_energy


    def T60_en_2_T60(self, T60_en):
        T60_en = torch.log10(T60_en + 1e-16) * 10
        T60_en = T60_en - T60_en[:, :, 0:1] + 60
        # T60_en_np = T60_en.cpu().numpy()
        mask = T60_en < 0
        # mask_np = mask.cpu().numpy()
        cumsum = mask.cumsum(dim=-1)
        # cumsum_np = cumsum.cpu().numpy()
        first_idx = (cumsum == 1).max(dim=-1).indices
        no_neg = mask.sum(dim=-1) == 0
        first_idx[no_neg] = -1   # 或者改成 x.size(-1)
        return (torch.max(first_idx, dim=-1)[0]).float() / 48000


    def sig_energy(self, sig):
        energy = (sig ** 2).sum(dim=(1, 2))
        return energy


    def to_dB(self, sig, w=10):
        return w * torch.log10(sig + 1e-16)

    def replace_invalid(self, x: torch.Tensor, value) -> torch.Tensor:
        """
        将张量中的 NaN 或 Inf 替换为指定值 (默认 160)
        """
        x = x.clone()  # 避免在原张量上就地修改
        mask = ~torch.isfinite(x)  # isfinite=True 表示正常值；取反后就是无效值
        x[mask] = value
        return x


    def RIR_MelSpectrogram_mask(self, x, typ):
        if typ == 'time':
            Mel_spec = self.Mel_spec_time
            k = 64
        elif typ == 'norm':
            Mel_spec = self.Mel_spec_norm
            k = 256
        else:
            Mel_spec = self.Mel_spec_freq
            k = 2048

        B, C, L = x.shape
        x = x.reshape(B * C, L)
        x = Mel_spec(x)
        x_mask = x.bool()
        x = torch.abs(x)
        x = torch.log10(x + 1e-6) * 20 + 120
        return x.float(), x_mask



def RIR_comparison(out_dir_dic, dic_path, RIR_len):
    try:
        with open(dic_path, 'rb') as f:
            err_dic= pickle.load(f)
    except Exception as e:
        err_dic = {}

    out_dir_list = list(out_dir_dic.keys())
    err_dic_new = {}

    for out_dir in out_dir_list:
        if out_dir_dic[out_dir]:
            print('\n\n', out_dir)
            batch_size = 32
            dataset = Dataset_evaluate(out_dir, RIR_len=RIR_len)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=32)
            eva = Evaluator(GPU_id=[0], batch_size=batch_size, RIR_len=RIR_len)
            error_mean_dic = eva.evaluate(dataloader)
            err_dic_new[out_dir] = error_mean_dic
            print(error_mean_dic)
        else:
            err_dic_new[out_dir] = err_dic[out_dir]
            print(err_dic_new[out_dir])

        with open(dic_path, 'wb') as f:
            pickle.dump(err_dic_new, f)
    with open(dic_path, 'wb') as f:
        pickle.dump(err_dic_new, f)
    return err_dic_new



def err_dec_2_excel(path):
    with open(path, 'rb') as f:
        err_dic = pickle.load(f)
    df = pd.DataFrame.from_dict(err_dic, orient="index")

    out_path = path.replace('.pickle', '.xlsx')
    df.to_excel(out_path)
    return



