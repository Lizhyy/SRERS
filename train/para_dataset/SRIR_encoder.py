import numpy as np
from Config import config
cfg = config()
import train.miscc.general_audio_processing as ap
import scipy.signal as scs



def sig_len_reshape(sig, target_length):
    """
        对输入信号进行长度重整（补零或截断），使其满足统一的目标长度。

    Parameters
    ----------
    sig : np.ndarray
        输入信号，可为单通道或多通道：
        - 若为一维数组，形状: (T,)
        - 若为二维数组，形状: (C, T)
          其中 C 为通道数，T 为当前采样点数。
    target_length : int
        目标信号长度（采样点数）。

    Returns
    -------
    np.ndarray
        调整长度后的信号：
        - 若 sig 为 1D → 输出形状 (target_length,)
        - 若 sig 为 2D → 输出形状 (C, target_length)
        - 数据类型与输入一致。
    """
    if np.ndim(sig) == 1:
        num_samples = len(sig)

        if num_samples < target_length:
            out_rir = np.zeros(target_length)
            out_rir[:num_samples] = sig
        else:
            out_rir = sig[:target_length]

    else:
        num_samples = len(sig[0])

        if num_samples < target_length:
            out_rir = np.zeros((len(sig), target_length))
            out_rir[:, :num_samples] = sig
        else:
            out_rir = sig[:, :target_length]
    return out_rir



def T0_calculate(srir):
    """
    计算每条 RIR（或 SRIR）信号的直达声起始点索引（T0）。

    Parameters
    ----------
    srir : np.ndarray
        输入的多通道房间脉冲响应信号数组。
        - 形状: (N, T)
            N 表示声源-接收点数量（或通道数）
            T 表示时域采样点数
        - 数据类型: float 或 np.float32 / np.float64
        - 每一行对应一条 RIR 信号。

    Returns
    -------
    np.ndarray
        每条 RIR 的起始点索引列表。
        - 形状: (N,)
        - 类型: int
        - 含义: 对应 RIR 信号中能量首次超过阈值（此处阈值为 0）的时间样本索引。
    """
    start_point_list = []
    for rir in srir:
        rir_en = np.sum(rir ** 2)
        rir_len = len(rir)
        start_point = 0
        s_en_th = rir_en * 0.001

        for i in range(0, rir_len):
            s_en = np.sum(rir[:i] ** 2)
            if s_en > s_en_th:
                start_point = i
                break
        start_point_list.append(start_point)
    return np.array(start_point_list)



def T60_calculate(srir):
    """
    计算每条 RIR（或 SRIR）信号的能量衰减 60 dB 对应的时间点索引（T60）。

    Parameters
    ----------
    srir : np.ndarray
        输入的多通道房间脉冲响应信号数组。
        - 形状: (N, T)
        - 数据类型: float 或 np.float32 / np.float64
        - 每一行对应一条 RIR 信号。

    Returns
    -------
    np.ndarray
        每条 RIR 的混响末尾索引列表。
        - 形状: (N,)
        - 类型: int
        - 含义: RIR 信号能量衰减至初始总能量的 1e-6 时的样本索引。
          （即对应 60 dB 衰减点。）
    """
    end_point_list = []
    for rir in srir:
        rir_en = np.sum(rir ** 2)
        rir_len = len(rir)
        end_point = rir_len - 1
        e_en_th = rir_en * 1e-6

        for i in range(rir_len - 1, 0, -1):
            e_en = np.sum(rir[i:] ** 2)
            if e_en > e_en_th:
                end_point = i
                break
        end_point_list.append(end_point)
    return np.array(end_point_list)



def SRIR_ori_wav_file_cut(srir_ori):
    """
    对原始 SRIR（Spatial Room Impulse Response）信号进行有效片段截取。
    该函数通过自动检测每条 RIR 的起点 (T0) 和 60 dB 衰减终点 (T60)，
    截取出有效声学区域（包含直达声、早期反射和主要混响部分），
    去除信号前后的静默区或无效段。

    Parameters
    ----------
    srir_ori : np.ndarray
        原始的多通道或多源 SRIR 信号矩阵。
        - 形状: (N, T)
            N : 声源-接收点组合数（或通道数）
            T : 时域采样点数
        - 数据类型: float32 或 float64
        - 每行代表一条独立的房间脉冲响应。

    Returns
    -------
    srir : np.ndarray
        截取后的 SRIR 信号矩阵，仅保留有效声学片段。
        - 形状: (N, T_eff)
            其中 T_eff = max(T60)
        - 数据类型: float32
    T60 : np.ndarray
        每条 SRIR 的有效结束采样点索引（对应能量衰减 60 dB）。
        - 形状: (N,)
        - 数据类型: int
    T0 : np.ndarray
        每条 SRIR 的起始采样点索引（对应直达声到达时间）。
        - 形状: (N,)
        - 数据类型: int
    """
    T0_int = T0_calculate(srir_ori)
    srir_cut_T0 = np.array(srir_ori[:, np.min(T0_int):]).astype('float32')
    T60 = T60_calculate(srir_cut_T0)
    srir = np.array(srir_cut_T0[:, :np.max(T60)]).astype('float32')
    return srir, T0_int, T60



def RIR_energy_norm(srir, gain=cfg.DATA.RIR_input.mch_gain, sig_cut=False):
    """
    对输入的 RIR 或 SRIR 信号进行能量归一化。

    Parameters
    ----------
    srir : np.ndarray
        输入 RIR 或 SRIR 信号。
        - 可为 1D 或 2D 数组：
          (T,) 或 (C, T)
          C 为通道数，T 为采样点数。
    gain : float
        归一化后的目标增益值（缩放系数），通常为配置项中的
        `cfg.DATA.RIR_input.mch_gain`。
    sig_cut : int or bool, optional
        若指定为整数，则在归一化前将信号裁剪或补零至该长度；
        若为 False（默认），不修改信号长度。

    Returns
    -------
    srir_norm : np.ndarray, srir_en : float
        归一化后的信号，与输入形状一致。
        当原信号能量为 0 时输出全零。
        原始信号的总能量（平方和或 RMS 值）。
    """
    if sig_cut:
        srir = sig_len_reshape(np.array(srir), sig_cut)
    srir_en = ap.sig_energy(srir, True)
    if srir_en != 0:
        srir_norm = srir / srir_en * gain
    else:
        srir_norm = srir * 0
    return srir_norm.astype('float32'), srir_en.astype('float32')



def SRIR_to_reverb(srir, LoR):
    """
    从 SRIR（Spatial Room Impulse Response）中分离出混响部分（Reverb），
    即通过去除早期反射 (LoR, Line of Reflection) 来获得后期混响响应。

    Parameters
    ----------
    srir : np.ndarray
        原始 SRIR 信号矩阵。
        - 形状: (C, T)
            C : 通道数（或声源-接收点组合数）
            T : 时域采样点数
        - 数据类型: float32 或 float64
        - 通常包含直达声、早期反射与晚期混响。

    LoR : np.ndarray
        早期反射（Line of Reflection）信号。
        - 形状应与 srir 一致或至少在时间维上匹配。
        - 表示在 SRIR 中被识别或估计出的早期反射成分。
        - 可以理解为直达声之后约 30–50 ms 内的响应。

    Returns
    -------
    reverb : np.ndarray
        提取出的混响部分（即 SRIR 去除早期反射后剩余的部分）。
        - 形状: (C, T)
        - 数据类型: 与输入一致
    """
    srir_len = len(srir[0])
    reverb = np.array(srir)
    if srir_len > 4096:
        reverb[:, :4096] = reverb[:, :4096] - 0.95 * LoR
    else:
        reverb = reverb - 0.95 * LoR[:, :srir_len]
    return reverb.astype('float32')



def f_bank_maker():
    """
    构建一组 FIR 带通滤波器组（filter bank），用于将 RIR 信号分解到不同频带。

    Returns
    -------
    f_bank : np.ndarray
        滤波器组矩阵，形状 (B, L)。
        - B : 频带数量（此处为 10 个）
        - L : 滤波器长度（signal_len = 48000）
        - 每一行是一个 FIR 滤波器系数序列。
    """
    f_bank = []
    signal_len = 48000
    fpass_list = [[1, 150], [150, 400], [400, 800], [800, 1300], [1300, 2000],
                  [2000, 3000], [3000, 6000], [6000, 10000], [10000, 16000], [16000, 23999]]
    for fpass in fpass_list:
        f = scs.firwin(signal_len, fpass, fs=48000, window=('kaiser', 9), pass_zero=False)
        f_bank.append(f)
    f_bank = np.array(f_bank)
    np.save("./filter_bank.npy", f_bank)
    return f_bank



def get_envelop(signal, point):
    """
    计算输入信号的平均能量包络（linear 或 log 均可进一步处理）。

    Parameters
    ----------
    signal : np.ndarray
        输入一维信号 (T,)
        - T : 采样点数
    point : int
        下采样后的包络点数。
        通常用于将长时信号压缩为低分辨率能量曲线。

    Returns
    -------
    ev : np.ndarray
        信号包络（线性能量均值），形状 (point,)
    """
    win_len = round(len(signal) / point + 0.5)
    ev_zeros = np.zeros(point * win_len)
    ev_zeros[:len(signal)] = signal
    ev_zeros = ev_zeros.reshape(-1, win_len)
    ev = np.mean(ev_zeros, axis=1)
    return ev.astype('float32')



def late_RIR_to_ev(reverb):
    """
        将输入的晚期混响信号（late RIR）转换为多频带能量包络矩阵。

    Parameters
    ----------
    RIR_late : np.ndarray
        晚期混响信号 (T,)
        - T : 采样点数
        - 通常为 SRIR 去除早期反射 (LoR) 后的残余混响部分。

    Returns
    -------
    RIR_ev_array : np.ndarray
        频带能量包络矩阵，形状 (B, N)
        - B : 频带数量（10）
        - N : 包络下采样点数（由 `get_envelop` 决定，此处为 64）
    """
    try:
        filter_bank = np.load("./filter_bank.npy")
    except Exception as e:
        filter_bank = f_bank_maker()

    filter_len = len(filter_bank[0])
    reverb_ev_array = []
    for reverb_mono in reverb:
        reverb_mono_ev_array = []
        for filter in filter_bank:
            bp_rir = scs.convolve(reverb_mono, filter, 'full')
            bp_rir = bp_rir[int(filter_len / 2): len(reverb_mono) + int(filter_len / 2)]
            ev = get_envelop(bp_rir ** 2, 64)
            ev_dB = np.log10(ev + 1e-12)
            ev_dB[ev_dB < -12] = -12
            ev_dB = (ev_dB + 12) / 6
            reverb_mono_ev_array.append(ev_dB)
        reverb_mono_ev_array = np.array(reverb_mono_ev_array)
        reverb_ev_array.append(reverb_mono_ev_array)
    return np.array(reverb_ev_array).astype('float32')



def SRIR_encoder_phase1(SRIR_ori, LoR_ori):
    SRIR_cut, T0_int, T60_int = SRIR_ori_wav_file_cut(SRIR_ori)
    LoR_cut = sig_len_reshape(LoR_ori[:, np.min(T0_int):], 4096)
    LoR_cut = LoR_cut.astype('float32')
    return SRIR_cut, LoR_cut, T0_int, T60_int



def SRIR_encoder_phase2(SRIR_cut, LoR_cut):
    reverb = SRIR_to_reverb(SRIR_cut, LoR_cut)
    SRIR_ER = sig_len_reshape(SRIR_cut, 4096)
    SRIR_ER_norm, SRIR_ER_en = RIR_energy_norm(SRIR_ER)
    reverb_ER = sig_len_reshape(reverb, 4096)
    reverb_ER_norm, reverb_ER_en = RIR_energy_norm(reverb_ER)
    reverb_norm, reverb_en = RIR_energy_norm(reverb)
    reverb_norm_ev = late_RIR_to_ev(reverb_norm)
    return (SRIR_ER_norm, SRIR_ER_en, reverb_ER_norm, reverb_ER_en,
            reverb, reverb_norm, reverb_en, reverb_norm_ev)

