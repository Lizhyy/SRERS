import os
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from CConfig import config
import function_package.Data_IO as dio
import train.para_dataset.SRIR_encoder as enc

cfg = config()
env_name = os.environ.get('CONDA_DEFAULT_ENV')


def Para_load_from_dic(data_cfg, RIR_para_dic, RIR_para_full_path):
    """Load one GSLR SRIR parameter dictionary and return model-ready arrays."""
    if data_cfg.INIT.Dataset != 'GSLR':
        raise ValueError('Minimal publish version only supports GSLR.')
    return Para_load_from_dic_GSLR(data_cfg, RIR_para_dic, RIR_para_full_path)


def Para_load_from_dic_GSLR(data_cfg, RIR_para_dic, RIR_para_full_path):
    LoR_t = data_cfg.INIT.IO_LoR_od
    IO_RIR_ER_type = data_cfg.INIT.IO_RIR_ER_type
    IO_RIR_out_ch = data_cfg.INIT.IO_RIR_out_ch

    try:
        if IO_RIR_ER_type == 'RIR':
            SRIR_ER_norm = RIR_para_dic['SRIR_ER_norm']
            SRIR_ER_en = RIR_para_dic['SRIR_ER_en']
        elif IO_RIR_ER_type == 'reverb':
            SRIR_ER_norm = RIR_para_dic[f'reverb_ER_norm_{LoR_t}']
            SRIR_ER_en = RIR_para_dic[f'reverb_ER_en_{LoR_t}']
        else:
            raise ValueError(f'Unsupported IO_RIR_ER_type: {IO_RIR_ER_type}')

        reverb_norm_ev = RIR_para_dic[f'reverb_norm_ev_{LoR_t}']
        if IO_RIR_out_ch == 1:
            SRIR_ER_norm = np.array([SRIR_ER_norm[0]])
            reverb_norm_ev = np.array([reverb_norm_ev[0]])
        elif IO_RIR_out_ch != 4:
            raise ValueError('Minimal publish version only supports 4-channel output.')

        reverb_energy_key = f'reverb_ER_en_{LoR_t}' if env_name == 'sre_n7' else f'reverb_en_{LoR_t}'
        para = np.array([
            np.max(RIR_para_dic['T60_int']) / 48000,
            SRIR_ER_en,
            RIR_para_dic[reverb_energy_key],
        ], dtype=np.float32)

        if LoR_t == 'od2':
            LoR_cut = RIR_para_dic['LoR_od2_cut']
        elif LoR_t == 'od1':
            LoR_cut = RIR_para_dic['LoR_od1_cut']
        else:
            raise ValueError(f'Unsupported IO_LoR_od: {LoR_t}')
        return SRIR_ER_norm, para, reverb_norm_ev, LoR_cut
    except Exception:
        print(f'Para_load_from_dic出错: {RIR_para_full_path}')
        return None


def SRIR_para_dic_loader(SRIR_para_full_path, tag_check, data_cfg=cfg, re_write=False):
    """Load or regenerate the od-specific GSLR parameter dictionary."""
    if data_cfg.INIT.Dataset != 'GSLR':
        raise ValueError('Minimal publish version only supports GSLR.')
    return SRIR_para_dic_loader_GSLR(SRIR_para_full_path, tag_check, data_cfg, re_write)


def SRIR_para_dic_loader_GSLR(SRIR_para_full_path, tag_check, data_cfg=cfg, re_write=False):
    if re_write:
        return para_full_2_od(SRIR_para_full_path, tag_check, data_cfg, re_write)

    IO_LoR_od = data_cfg.INIT.IO_LoR_od
    if IO_LoR_od == 'od1':
        SRIR_para_od_path = SRIR_para_full_path.replace('/para_full/', '/para_od1/')
    elif IO_LoR_od == 'od2':
        SRIR_para_od_path = SRIR_para_full_path.replace('/para_full/', '/para_od2/')
    else:
        raise ValueError(f'Unsupported IO_LoR_od: {IO_LoR_od}')

    try:
        SRIR_para_od_dic = dio.load_pickle(SRIR_para_od_path)
    except Exception:
        print(f'无法打开para: {SRIR_para_od_path}')
        SRIR_para_od_dic = para_full_2_od(SRIR_para_full_path, tag_check, data_cfg, re_write)

    try:
        if SRIR_para_od_dic['tag'] != tag_check:
            print(f'当前tag：{SRIR_para_od_dic["tag"]}，设定tag：{tag_check}')
            SRIR_para_od_dic = para_full_2_od(SRIR_para_full_path, tag_check, data_cfg, re_write)
    except Exception:
        print(f'缺少tag的key: {SRIR_para_full_path}')
        SRIR_para_od_dic = para_full_2_od(SRIR_para_full_path, tag_check, data_cfg, re_write)

    if env_name != 'sre_n7':
        LoR_t = data_cfg.INIT.IO_LoR_od
        if f'reverb_en_{LoR_t}' not in SRIR_para_od_dic:
            print(f'缺少reverb_en的key: {SRIR_para_full_path}')
            SRIR_para_od_dic = para_full_2_od(SRIR_para_full_path, tag_check, data_cfg, re_write)

    return SRIR_para_od_dic


def para_full_2_od(SRIR_para_full_path, tag_check, data_cfg=cfg, re_write=False):
    """Convert a GSLR para_full pickle into od1/od2 lightweight pickles."""
    if re_write:
        SRIR_para_full_dic = para_full_remake_from_old_para(SRIR_para_full_path, tag_check)
    else:
        try:
            SRIR_para_full_dic = dio.load_pickle(SRIR_para_full_path)
        except Exception:
            print(f'无法打开para_full: {SRIR_para_full_path}')
            SRIR_para_full_dic = para_full_remake_from_old_para(SRIR_para_full_path, tag_check)

    try:
        if SRIR_para_full_dic['tag'] != tag_check:
            print(f'当前tag：{SRIR_para_full_dic["tag"]}，设定tag：{tag_check}')
            SRIR_para_full_dic = para_full_remake_from_old_para(SRIR_para_full_path, tag_check)
    except Exception:
        print(f'缺少tag的key: {SRIR_para_full_path}')
        SRIR_para_full_dic = para_full_remake_from_old_para(SRIR_para_full_path, tag_check)

    SRIR_para_od1_path = SRIR_para_full_path.replace('/para_full/', '/para_od1/')
    SRIR_para_od2_path = SRIR_para_full_path.replace('/para_full/', '/para_od2/')
    os.makedirs(os.path.dirname(SRIR_para_od1_path), exist_ok=True)
    os.makedirs(os.path.dirname(SRIR_para_od2_path), exist_ok=True)

    od1_keys = ['SRIR_ER_norm', 'SRIR_ER_en', 'reverb_ER_norm_od1', 'reverb_ER_en_od1',
                'reverb_norm_ev_od1', 'T60_int', 'reverb_en_od1', 'LoR_od1_cut', 'tag']
    od2_keys = ['SRIR_ER_norm', 'SRIR_ER_en', 'reverb_ER_norm_od2', 'reverb_ER_en_od2',
                'reverb_norm_ev_od2', 'T60_int', 'reverb_en_od2', 'LoR_od2_cut', 'tag']

    SRIR_para_od1_dic = para_full_reshape(SRIR_para_full_dic, od1_keys)
    SRIR_para_od2_dic = para_full_reshape(SRIR_para_full_dic, od2_keys)
    dio.save_pickle(SRIR_para_od1_dic, SRIR_para_od1_path)
    dio.save_pickle(SRIR_para_od2_dic, SRIR_para_od2_path)

    if data_cfg.INIT.IO_LoR_od == 'od1':
        return SRIR_para_od1_dic
    if data_cfg.INIT.IO_LoR_od == 'od2':
        return SRIR_para_od2_dic
    raise ValueError(f'Unsupported IO_LoR_od: {data_cfg.INIT.IO_LoR_od}')


def para_full_remake_from_old_para(SRIR_para_full_path, tag_check):
    """Regenerate a GSLR para_full pickle from source wav files."""
    result = para_full_remake_from_ori_file(SRIR_para_full_path)
    if result is None:
        return None
    SRIR_cut, T0_int, T60_int, LoR_od1_cut, LoR_od2_cut = result

    SRIR_para_full_dic_new = {
        'SRIR_cut': SRIR_cut.astype('float32'),
        'T0_int': T0_int,
        'T60_int': T60_int,
        'LoR_od1_cut': LoR_od1_cut.astype('float32'),
        'LoR_od2_cut': LoR_od2_cut.astype('float32'),
        'tag': tag_check,
    }

    (SRIR_ER_norm, SRIR_ER_en, reverb_ER_norm_od1, reverb_ER_en_od1, _, _,
     reverb_en_od1, reverb_norm_ev_od1) = enc.SRIR_encoder_phase2(SRIR_cut, LoR_od1_cut)
    SRIR_para_full_dic_new['SRIR_ER_norm'] = SRIR_ER_norm.astype('float32')
    SRIR_para_full_dic_new['SRIR_ER_en'] = SRIR_ER_en.astype('float32')
    SRIR_para_full_dic_new['reverb_ER_norm_od1'] = reverb_ER_norm_od1.astype('float32')
    SRIR_para_full_dic_new['reverb_ER_en_od1'] = reverb_ER_en_od1.astype('float32')
    SRIR_para_full_dic_new['reverb_en_od1'] = reverb_en_od1.astype('float32')
    SRIR_para_full_dic_new['reverb_norm_ev_od1'] = reverb_norm_ev_od1.astype('float32')

    (_, _, reverb_ER_norm_od2, reverb_ER_en_od2, _, _,
     reverb_en_od2, reverb_norm_ev_od2) = enc.SRIR_encoder_phase2(SRIR_cut, LoR_od2_cut)
    SRIR_para_full_dic_new['reverb_ER_norm_od2'] = reverb_ER_norm_od2.astype('float32')
    SRIR_para_full_dic_new['reverb_ER_en_od2'] = reverb_ER_en_od2.astype('float32')
    SRIR_para_full_dic_new['reverb_en_od2'] = reverb_en_od2.astype('float32')
    SRIR_para_full_dic_new['reverb_norm_ev_od2'] = reverb_norm_ev_od2.astype('float32')

    dio.save_pickle(SRIR_para_full_dic_new, SRIR_para_full_path)
    return SRIR_para_full_dic_new


def para_full_remake_from_ori_file(SRIR_para_full_path):
    SRIR_wav_path = SRIR_para_full_path.replace('/para_full/SRIR_', '/SRIR2/SRIR_').replace('.pickle', '.wav')
    LoR_od2_path = SRIR_wav_path.replace('/SRIR2/SRIR_', '/ER_Nd/ER_Nd_')
    LoR_od1_path = SRIR_wav_path.replace('/SRIR2/SRIR_', '/Ely_Reflc/Ely_Reflc_')
    try:
        SRIR_ori = dio.load(SRIR_wav_path)
        LoR_od2_ori = dio.load(LoR_od2_path)
        LoR_od1_ori = dio.load(LoR_od1_path)
        SRIR_cut, LoR_od1_cut, T0_int, T60_int = enc.SRIR_encoder_phase1(SRIR_ori, LoR_od1_ori)
        LoR_od2_cut = enc.sig_len_reshape(LoR_od2_ori[:, np.min(T0_int):], 4096)
        return SRIR_cut, T0_int, T60_int, LoR_od1_cut, LoR_od2_cut
    except Exception:
        print('源文件损坏或缺失，无法重新生成GSLR参数文件。')
        return None


def para_full_reshape(para_full_dic, key_list):
    return {key: para_full_dic[key] for key in key_list}
