from Config import config
import os
cfg = config()
import train.para_dataset.SRIR_encoder as enc
import numpy as np
import train.miscc.Data_IO as dio

env_name = os.environ.get("CONDA_DEFAULT_ENV")




def Para_load_from_dic(data_cfg, RIR_para_dic, RIR_para_full_path):
    LoR_t = data_cfg.INIT.IO_LoR_od
    IO_RIR_ER_type = data_cfg.INIT.IO_RIR_ER_type
    try:
        if IO_RIR_ER_type == 'RIR':
            SRIR_ER_norm = RIR_para_dic['SRIR_ER_norm']
            SRIR_ER_en = RIR_para_dic['SRIR_ER_en']
        elif IO_RIR_ER_type == 'reverb':
            SRIR_ER_norm = RIR_para_dic['reverb_ER_norm_{}'.format(LoR_t)]
            SRIR_ER_en = RIR_para_dic['reverb_ER_en_{}'.format(LoR_t)]

        reverb_norm_ev = RIR_para_dic['reverb_norm_ev_{}'.format(LoR_t)]
        para = np.array([np.max(RIR_para_dic['T60_int']) / 48000, SRIR_ER_en,
                         RIR_para_dic['reverb_en_{}'.format(LoR_t)]])

        if LoR_t == 'od2':
            LoR_cut = RIR_para_dic['LoR_od2_cut']
        elif LoR_t == 'od1':
            LoR_cut = RIR_para_dic['LoR_od1_cut']
        return SRIR_ER_norm, para, reverb_norm_ev, LoR_cut
    except Exception as e:
        print('Para_load_from_dic出错，估计问题比较复杂:{}'.format(RIR_para_full_path))
        return None




def SRIR_para_dic_loader(SRIR_para_full_path, tag_check, data_cfg=cfg, re_write=False):
    if re_write:
        SRIR_para_od_dic = para_full_2_od(SRIR_para_full_path, tag_check, data_cfg, re_write)
    else:
        IO_LoR_od = data_cfg.INIT.IO_LoR_od
        if IO_LoR_od == 'od1':
            SRIR_para_od_path = SRIR_para_full_path.replace('/para_full/', '/para_od1/')
        elif IO_LoR_od == 'od2':
            SRIR_para_od_path = SRIR_para_full_path.replace('/para_full/', '/para_od2/')
        else:
            print('cfg.INIT.IO_LoR_od error : {}'.format(SRIR_para_full_path))
            return None

        try:
            SRIR_para_od_dic = dio.load_pickle(SRIR_para_od_path)
        except Exception as e:
            print('无法打开para：{}'.format(SRIR_para_od_path))
            SRIR_para_od_dic = para_full_2_od(SRIR_para_full_path, tag_check, data_cfg, re_write)

        try:
            if SRIR_para_od_dic['tag'] != tag_check:
                print('当前tag：{}，设定tag：{}'.format(SRIR_para_od_dic['tag'], tag_check))
                print('tag不符合，说明para版本不对：{}'.format(SRIR_para_full_path))
                SRIR_para_od_dic = para_full_2_od(SRIR_para_full_path, tag_check, data_cfg, re_write)
        except Exception as e:
            print('缺少tag的key，字典生成代码可能有错：{}'.format(SRIR_para_full_path))
            SRIR_para_od_dic = para_full_2_od(SRIR_para_full_path, tag_check, data_cfg, re_write)

        if not env_name == 'sre_n7':
            try:
                LoR_t = data_cfg.INIT.IO_LoR_od
                a = SRIR_para_od_dic['reverb_en_{}'.format(LoR_t)]
            except Exception as e:
                print('缺少reverb_en的key，还没来得及处理：{}'.format(SRIR_para_full_path))
                SRIR_para_od_dic = para_full_2_od(SRIR_para_full_path, tag_check, data_cfg, re_write)

    return SRIR_para_od_dic




def para_full_2_od(SRIR_para_full_path, tag_check, data_cfg=cfg, re_write=False):
    if re_write:
        SRIR_para_full_dic = para_full_remake_from_old_para(SRIR_para_full_path, tag_check)
    else:
        SRIR_para_od1_path = SRIR_para_full_path.replace('/para_full/', '/para_od1/')
        SRIR_para_od2_path = SRIR_para_full_path.replace('/para_full/', '/para_od2/')

        try:
            SRIR_para_full_dic = dio.load_pickle(SRIR_para_full_path)
        except Exception as e:
            print('无法打开para_full：{}'.format(SRIR_para_full_path))
            SRIR_para_full_dic = para_full_remake_from_old_para(SRIR_para_full_path, tag_check)

    try:
        if SRIR_para_full_dic['tag'] != tag_check:
            print('当前tag：{}，设定tag：{}'.format(SRIR_para_full_dic['tag'], tag_check))
            print('tag不符合，说明para_full版本不对：{}'.format(SRIR_para_full_path))
            SRIR_para_full_dic = para_full_remake_from_old_para(SRIR_para_full_path, tag_check)
    except Exception as e:
        print('缺少tag的key，字典生成代码可能有错：{}'.format(SRIR_para_full_path))
        SRIR_para_full_dic = para_full_remake_from_old_para(SRIR_para_full_path, tag_check)

    ky1_list = ['SRIR_ER_norm', 'SRIR_ER_en', 'reverb_ER_norm_od1', 'reverb_ER_en_od1',
                'reverb_norm_ev_od1', 'T60_int', 'reverb_en_od1', 'LoR_od1_cut', 'tag']
    SRIR_para_od1_dic = para_full_reshape(SRIR_para_full_dic, ky1_list)
    dio.save_pickle(SRIR_para_od1_dic, SRIR_para_od1_path)

    ky2_list = ['SRIR_ER_norm', 'SRIR_ER_en', 'reverb_ER_norm_od2', 'reverb_ER_en_od2',
                'reverb_norm_ev_od2', 'T60_int', 'reverb_en_od2', 'LoR_od2_cut', 'tag']
    SRIR_para_od2_dic = para_full_reshape(SRIR_para_full_dic, ky2_list)
    dio.save_pickle(SRIR_para_od2_dic, SRIR_para_od2_path)
    IO_LoR_od = data_cfg.INIT.IO_LoR_od
    if IO_LoR_od == 'od1':
        return SRIR_para_od1_dic
    elif IO_LoR_od == 'od2':
        return SRIR_para_od2_dic
    else:
        print('cfg.INIT.IO_LoR_od error : {}'.format(SRIR_para_full_path))
        return None



def para_full_remake_from_old_para(SRIR_para_full_path, tag_check):
    (SRIR_cut, T0_int, T60_int, LoR_od1_cut,
     LoR_od2_cut) = para_full_remake_from_ori_file(SRIR_para_full_path)

    SRIR_para_full_dic_new = {}
    SRIR_para_full_dic_new['SRIR_cut'] = SRIR_cut.astype('float32')
    SRIR_para_full_dic_new['T0_int'] = T0_int
    SRIR_para_full_dic_new['T60_int'] = T60_int
    SRIR_para_full_dic_new['LoR_od2_cut'] = LoR_od2_cut.astype('float32')
    SRIR_para_full_dic_new['LoR_od1_cut'] = LoR_od1_cut.astype('float32')
    (SRIR_ER_norm, SRIR_ER_en, reverb_ER_norm_od1, reverb_ER_en_od1, reverb_od1, reverb_norm_od1,
     reverb_en_od1, reverb_norm_ev_od1) = enc.SRIR_encoder_phase2(SRIR_cut, LoR_od1_cut)

    SRIR_para_full_dic_new['SRIR_ER_norm'] = SRIR_ER_norm.astype('float32')
    SRIR_para_full_dic_new['SRIR_ER_en'] = SRIR_ER_en.astype('float32')
    SRIR_para_full_dic_new['reverb_ER_norm_od1'] = reverb_ER_norm_od1.astype('float32')
    SRIR_para_full_dic_new['reverb_ER_en_od1'] = reverb_ER_en_od1.astype('float32')
    # SRIR_para_full_dic_new['reverb_od1'] = reverb_od1.astype('float32')
    # SRIR_para_full_dic_new['reverb_norm_od1'] = reverb_norm_od1.astype('float32')
    SRIR_para_full_dic_new['reverb_en_od1'] = reverb_en_od1.astype('float32')
    SRIR_para_full_dic_new['reverb_norm_ev_od1'] = reverb_norm_ev_od1.astype('float32')

    (SRIR_ER_norm, SRIR_ER_en, reverb_ER_norm_od2, reverb_ER_en_od2, reverb_od2, reverb_norm_od2,
     reverb_en_od2, reverb_norm_ev_od2) = enc.SRIR_encoder_phase2(SRIR_cut, LoR_od2_cut)

    SRIR_para_full_dic_new['reverb_ER_norm_od2'] = reverb_ER_norm_od2.astype('float32')
    SRIR_para_full_dic_new['reverb_ER_en_od2'] = reverb_ER_en_od2.astype('float32')
    # SRIR_para_full_dic_new['reverb_od2'] = reverb_od2.astype('float32')
    # SRIR_para_full_dic_new['reverb_norm_od2'] = reverb_norm_od2.astype('float32')
    SRIR_para_full_dic_new['reverb_en_od2'] = reverb_en_od2.astype('float32')
    SRIR_para_full_dic_new['reverb_norm_ev_od2'] = reverb_norm_ev_od2.astype('float32')

    SRIR_para_full_dic_new['tag'] = tag_check
    dio.save_pickle(SRIR_para_full_dic_new, SRIR_para_full_path)
    return SRIR_para_full_dic_new



def para_full_remake_from_ori_file(SRIR_para_full_path):
    SRIR_wav_path = SRIR_para_full_path.replace('/para_full/SRIR_', '/SRIR2/SRIR_')
    SRIR_wav_path = SRIR_wav_path.replace('.pickle', '.wav')
    LoR_od2_path = SRIR_wav_path.replace('/SRIR2/SRIR_', '/ER_Nd/ER_Nd_')
    LoR_od1_path = SRIR_wav_path.replace('/SRIR2/SRIR_', '/Ely_Reflc/Ely_Reflc_')
    try:
        SRIR_ori = dio.load(SRIR_wav_path)
        LoR_od2_ori = dio.load(LoR_od2_path)
        LoR_od1_ori = dio.load(LoR_od1_path)

        SRIR_cut, LoR_od1_cut, T0_int, T60_int = enc.SRIR_encoder_phase1(SRIR_ori, LoR_od1_ori)
        LoR_od2_cut = enc.sig_len_reshape(LoR_od2_ori[:, np.min(T0_int):], 4096)
        return SRIR_cut, T0_int, T60_int, LoR_od1_cut, LoR_od2_cut
    except Exception as e:
        print('源文件损坏，重新跑pygsound吧哈哈哈哈哈哈哈')
        return None



def para_full_reshape(para_full_dic,
                      key_list = [
                          'RIR_ER', 'RIR_ER_en', 'reverb_ER', 'reverb_ER_en',
                          'reverb_ev', 'T60', 'reverb_en', 'RIR_od1', 'RIR_od2', 'tag']
                      ):
    para_dic = {}
    for key in key_list:
        para_dic[key] = para_full_dic[key]
    return para_dic


