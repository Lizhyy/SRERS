import os
import numpy as np
from train.miscc.Data_IO import output as output_RIR
from Config import config
import pickle
from train.para_dataset.SRIR_decoder import SRERS_decoder
cfg = config()
from time import time


def dic_load(path):
    with open(path, 'rb') as f:
        dic = pickle.load(f)
    return dic


def dic_save(path, dic):
    with open(path, 'wb') as f:
        pickle.dump(dic, f)
    return dic



def dic_update(keyy, data, dic_path, wav_path):
    try:
        dic = dic_load(dic_path)
    except Exception as e:
        dic = {}

    dic[keyy] = data

    try:
        RIR = dic['ER']
        para = dic['para']
        reverb_ev = dic['reverb_ev']
        LoR = dic['LoR']
        dic_save(dic_path, dic)
        SRIR_full = SRERS_decoder(RIR, para, reverb_ev, np.zeros([4, 4096]))
        output_RIR(SRIR_full, wav_path, show_info=False)
    except Exception as e:
        dic_save(dic_path, dic)
    return




def ori_path_decoder(rir_ori_path_No):
    if cfg.INIT.Dataset == 'GSLR':
        rir_ori_path = 'House_{}/SRIR/SRIR_S{}_L{}'.format(
                                str(rir_ori_path_No[-3]).zfill(4),
                                str(rir_ori_path_No[-2]),
                                str(rir_ori_path_No[-1]).zfill(4)
                            )
    else:
        rir_ori_path = 'House_{}/SRIR2/L{}_R{}'.format(
                                str(rir_ori_path_No[-3]).zfill(4),
                                str(rir_ori_path_No[-2]),
                                str(rir_ori_path_No[-1]).zfill(4)
                            )
    rir_ori_folder_list = rir_ori_path.split('/')
    room_name = rir_ori_folder_list[-3]
    rir_name = rir_ori_folder_list[-1]
    return rir_ori_path, room_name, rir_name




def RIR_path_maker(rir_ori_path_No, out_path, IDi=0, IDj=0):
    if cfg.INIT.Dataset == 'GWA':
        room_name = str(IDi).zfill(5)
        rir_name = str(IDj).zfill(6)
        # 原始RIR的路径
        ori_dic_load_path = rir_ori_path_No

        out_folder_path = os.path.join(out_path, room_name)
        if not os.path.exists(out_folder_path):
            os.makedirs(out_folder_path)

        out_dic_folder_path = os.path.join(out_folder_path, 'dic')
        if not os.path.exists(out_dic_folder_path):
            os.makedirs(out_dic_folder_path)

        out_dic_room_folder_path = os.path.join(out_dic_folder_path, room_name)
        if not os.path.exists(out_dic_room_folder_path):
            os.makedirs(out_dic_room_folder_path)

        ori_wav_out_path = os.path.join(out_folder_path, rir_name + '_ori.wav')
        real_wav_out_path = os.path.join(out_folder_path, rir_name + '_real.wav')
        infe_wav_out_path = os.path.join(out_folder_path, rir_name + '_infe.wav')
        real_dic_out_path = os.path.join(out_dic_room_folder_path, rir_name + '_real.pickle')
        infe_dic_out_path = os.path.join(out_dic_room_folder_path, rir_name + '_infe.pickle')

    else:
        _, room_name, rir_name = ori_path_decoder(rir_ori_path_No)
        # 原始RIR的路径
        ori_dic_load_path = os.path.join(cfg.DATA.test_dataset_path, room_name,
                                          'para_full', rir_name + '.pickle')

        out_folder_path = os.path.join(out_path, room_name)
        if not os.path.exists(out_folder_path):
            os.makedirs(out_folder_path)

        out_dic_folder_path = os.path.join(out_folder_path, 'dic')
        if not os.path.exists(out_dic_folder_path):
            os.makedirs(out_dic_folder_path)

        out_dic_room_folder_path = os.path.join(out_dic_folder_path, room_name)
        if not os.path.exists(out_dic_room_folder_path):
            os.makedirs(out_dic_room_folder_path)

        ori_wav_out_path = os.path.join(out_folder_path, rir_name + '_ori.wav')
        real_wav_out_path = os.path.join(out_folder_path, rir_name + '_real.wav')
        infe_wav_out_path = os.path.join(out_folder_path, rir_name + '_infe.wav')
        real_dic_out_path = os.path.join(out_dic_room_folder_path, rir_name + '_real.pickle')
        infe_dic_out_path = os.path.join(out_dic_room_folder_path, rir_name + '_infe.pickle')

    return (ori_dic_load_path, ori_wav_out_path, real_wav_out_path,
            real_dic_out_path, infe_dic_out_path, infe_wav_out_path)



def SRERS_separate_RIR_decoder(real_RIR_list, infe_RIR_ts, rir_ori_path_No_ts, LoR_wav_ts):
    decoder_list = cfg.INIT.RIR_decoder.split('-')
    de_type = decoder_list[0]
    if de_type == 'ER':
        keyy = 'ER'
        real_RIR_ts = real_RIR_list[0]
    elif de_type == 'PA':
        keyy = 'para'
        real_RIR_ts = real_RIR_list[1]
    elif de_type == 'LR':
        keyy = 'reverb_ev'
        real_RIR_ts = real_RIR_list[2]

    show_info_set = False
    real_RIR_array = np.array(real_RIR_ts.to("cpu").detach())
    infe_RIR_array = np.array(infe_RIR_ts.to("cpu").detach())
    rir_ori_path_No_array = np.array(rir_ori_path_No_ts.to("cpu").detach())
    LoR_wav_array = np.array(LoR_wav_ts.to("cpu").detach())
    for real_RIR, infe_RIR, rir_ori_path_No, LoR_wav in zip(real_RIR_array, infe_RIR_array,
                                                            rir_ori_path_No_array, LoR_wav_array):
        # print(00, real_RIR, infe_RIR, '\n')
        (ori_dic_load_path, ori_wav_out_path, real_wav_out_path, real_dic_out_path,
         infe_dic_out_path, infe_wav_out_path) = RIR_path_maker(rir_ori_path_No)

        # if 1:
        if not os.path.exists(ori_wav_out_path):
            ori_RIR_dic = dic_load(ori_dic_load_path)
            ori_SRIR_wav = ori_RIR_dic['RIR']
            ori_LoR_wav = ori_RIR_dic['RIR_od2']
            min_lin = min(len(ori_SRIR_wav[0]), len(ori_LoR_wav[0]))
            if not cfg.TEST.with_LoR:
                ori_SRIR_wav[:, :min_lin] = ori_SRIR_wav[:, :min_lin] - ori_LoR_wav[:, :min_lin]
            output_RIR(ori_SRIR_wav, ori_wav_out_path, show_info=show_info_set)

        dic_update(keyy, real_RIR, real_dic_out_path, real_wav_out_path)
        dic_update('LoR', LoR_wav, real_dic_out_path, real_wav_out_path)
        if de_type == 'PA':
            infe_RIR[1:] = 10 ** ((infe_RIR[1:] * 2) - 3)
        dic_update(keyy, infe_RIR, infe_dic_out_path, infe_wav_out_path)
        dic_update('LoR', LoR_wav, infe_dic_out_path, infe_wav_out_path)
        # print(22, real_RIR, infe_RIR , '\n')
        a = 1
    return




def SRERS_full_RIR_decoder(real_ERs, real_LoRs, real_para, real_LR, fake_ER,
                           fake_para, fake_LR, RIR_path, out_folder_path, rank=0):
    real_ERs_array = real_ERs.to("cpu").detach().numpy()
    real_LoRs_array = real_LoRs.to("cpu").detach().numpy()
    real_para_array = real_para.to("cpu").detach().numpy()
    real_LRs_array = real_LR.to("cpu").detach().numpy()
    fake_ERs_array = fake_ER.to("cpu").detach().numpy()
    fake_para_array = fake_para.to("cpu").detach().numpy()
    fake_para_array[:, 1:] = 10 **(fake_para_array[:, 1:] - 3)
    fake_LRs_array = fake_LR.to("cpu").detach().numpy()
    RIR_path_No_array = RIR_path.to("cpu").detach().numpy()

    batch_len = len(real_ERs)
    dec_time = 0
    for i in range(batch_len):
        (ori_dic_load_path, ori_wav_out_path, real_wav_out_path, real_dic_out_path,
         infe_dic_out_path, infe_wav_out_path) = RIR_path_maker(RIR_path_No_array[i],
                                                                out_folder_path, rank, i)
        if cfg.INIT.Dataset == 'GWA':
            ori_SRIR_wav = np.array(ori_dic_load_path[:round(48000 * real_para_array[i][0])])
            LoR_wav = real_LoRs_array[i] * 0
        else:
            ori_RIR_dic = dic_load(ori_dic_load_path)
            ori_SRIR_wav = ori_RIR_dic['SRIR_cut']
            LoR_wav = real_LoRs_array[i]

        if not cfg.TEST.with_LoR:
            try:
                ori_SRIR_wav[:, :4096] = ori_SRIR_wav[:, :4096] - 0.95 * LoR_wav
            except:
                ori_SRIR_wav = ori_SRIR_wav - 0.95 *  LoR_wav[:, :len(ori_SRIR_wav[0])]

        real_SRIR_wav = SRERS_decoder(real_ERs_array[i], real_para_array[i],
                                      real_LRs_array[i], LoR_wav)
        start_t = time()
        fake_SRIR_wav = SRERS_decoder(fake_ERs_array[i], fake_para_array[i],
                                      fake_LRs_array[i], LoR_wav)
        end_t = time()
        t = end_t - start_t
        dec_time += t


        if cfg.INIT.Dataset == 'GWA':
            output_RIR(ori_SRIR_wav, ori_wav_out_path, show_info=False)
            output_RIR(real_SRIR_wav[0], real_wav_out_path, show_info=False)
            output_RIR(fake_SRIR_wav[0], infe_wav_out_path, show_info=False)
        else:
            output_RIR(ori_SRIR_wav, ori_wav_out_path, show_info=False)
            output_RIR(real_SRIR_wav, real_wav_out_path, show_info=False)
            output_RIR(fake_SRIR_wav, infe_wav_out_path, show_info=False)
        a = 1
    return dec_time / (batch_len + 1)




def RIR_para_decoder(para_path, output_path):
    room_folder_list = os.listdir(para_path)
    j = 1
    total_time = 0
    for room_folder_name in room_folder_list:
        in_room_folder_path = os.path.join(para_path, room_folder_name)
        # out_room_folder_path = os.path.join(output_path, room_folder_name)
        # if not os.path.exists(out_room_folder_path):
        #     os.makedirs(out_room_folder_path)

        for para_file in os.listdir(in_room_folder_path):
            in_para_path = os.path.join(in_room_folder_path, para_file)
            # wav_file = para_file.replace('.pickle', '.wav')
            # out_wav_path = os.path.join(out_room_folder_path, wav_file)
            # if os.path.exists(out_wav_path):
            #     continue
            # else:
            #     print(out_wav_path)
            rir_para = dic_load(in_para_path)
            ER = rir_para['ER']
            para = rir_para['para']
            reverb_ev = rir_para['reverb_ev']
            LoR = rir_para['LoR']
            start_t = time()
            SRIR_full = SRERS_decoder(ER, para, reverb_ev, np.zeros([4, 4096]))
            end_t = time()
            t = end_t - start_t
            total_time += t
            print(total_time / j)
            j += 1
    return

            # output_RIR(SRIR_full, out_wav_path)



