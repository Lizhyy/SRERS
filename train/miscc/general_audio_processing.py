"""General audio utilities used by SRERS helper scripts."""

import numpy as np
import librosa as lib
import train.miscc.Data_IO as dio
import scipy.fft as scf
import scipy.signal as scs
from scipy.io.wavfile import write
sr=48000


def audio_reshape(aud, dim2=True, max_ch=12):
    """
    检测音频是否满足array(声道，声压)的格式，若不是，转置
    :param aud: array，list均可
    :param dim2: 若为单声道，是否要转换成(1,n)的形式
    :param max_ch: 默认的最大声道数，默认取12声道
    :return: 满足(声道，采样点)的音频
    """
    aud = np.array(aud)
    if np.ndim(aud) == 1:
        print('1 channel')
        if dim2:
            return aud.reshape(1, -1)
        else:
            return aud
    else:
        if (aud.shape[0] > max_ch) & (aud.shape[0] > aud.shape[1]):
            return aud.T
        else:
            return aud



def audio_gain(audio, dB, resh=False):
    """
    Adjusting the audio's loudness level.
    :param audio: audio, array (channel, len)
    :param dB: dB gain, float
    :param resh: auto transpose if needed, bool
    :return: audio, array (channel, len)
    """
    gain = 10 ** (dB / 20)
    audio = np.array(audio)
    if np.ndim(audio) > 1:
        if (audio.shape[0] > 100) & (audio.shape[0] > audio.shape[1]):
            print('waring: audio may need transpose')
            if resh:
                audio = audio.T
    a = audio * gain
    return a


def sig_energy(sig, sq=False):
    """
    求得信号的总能量
    :param sig: 输入信号，array
    :param sq: 是否开方
    :return: 能量值，float
    """
    if sq:
        return np.sqrt(np.sum(np.square(np.abs(sig))))
    else:
        return np.sum(np.square(np.abs(sig)))


def audio_norm(audio, norn=0):
    """
    将音频标准化，设定最大值，默认为0dB
    :param audio: audio, array (channel, len)
    :param norn: gain num, float
    :return: audio, array (channel, len)
    """
    audio = audio / np.max(np.abs(audio))
    return audio_gain(audio, norn)


def stereo2surround(aud, full_stereo=True, norm=True):
    """
    stereo to 5.1 surround channel
    :param aud: audio, array (channel, len)
    :param full_stereo: ues full stereo or 5.1 panning, bool
    :param norm: norm to 0dB or not, boll
    :return: 5.1 channel audio, array (channel, len)
    """
    if np.ndim(aud) == 2:
        l = aud[0, :]
        r = aud[1, :]
        C = 0.5 * (l + r)
        LEF = np.zeros(len(l))
        if full_stereo:
            au51 = np.array([l, r, C, LEF, l, r])
        else:
            L = 0.8 * l + 0.25 * C
            R = 0.8 * r + 0.25 * C
            Ls = (l - C) * 1 + l * 0.5
            Rs = (r - C) * 1 + r * 0.5
            s = np.max(np.abs(np.array([Ls, Rs])))
            if s > 1:
                print('!')
            au51 = np.array([L, R, C, LEF, Ls, Rs])
    elif np.ndim(aud) == 1:
        au51 = np.array([aud, aud, aud, np.zeros(len(aud)), aud, aud])
    else:
        au51 = 0
        print('audio dim not support')
    if norm:
        return audio_norm(au51)
    else:
        return au51


def stereo_mix(audio, mix_rate=0, demix_rate=0):
    """
    改变双声道音乐的分离度
    :param audio: audio, array (channel, len)
    :param mix_rate: 分离度，0-1 float
    :param demix_rate: 去分离度，0-1 float
    :return: audio, array (channel, len)
    """
    if np.ndim(audio) == 1:
        audio = np.array([audio, audio])
    if audio.shape[0] != 2:
        audio = audio.T
        print('Input audio transpose')
    mix = (audio[0] + audio[1]) / 2
    mix = np.array([mix, mix])
    demix = (audio[0] - audio[1]) / 2
    demix = np.array([demix, -1*demix])
    if mix_rate != 0:
        audio_re = audio * (1-mix_rate) + mix * mix_rate
    elif demix_rate != 0:
        audio_re = audio * (1-demix_rate) + demix * demix_rate
    else:
        audio_re = audio
    return audio_norm(audio_re)


def A_weight_power(aud_or, sr=48000):
    """
    求输入音频A计权下的响度（本函数的绝对值可能有误，一般用于响度对齐而不是直接读数）
    :param aud_or: 输入音频
    :param sr: 采样率
    :return: 响度值
    """
    A_list = []
    fram = round(sr / 40)
    if np.ndim(aud_or) == 1:
        aud_or = aud_or.reshape(1, -1)
    for aud_or_mono in aud_or:
        aud_a = np.zeros(round(len(aud_or_mono) / fram +0.5)*fram)
        aud_a[:len(aud_or_mono)] = aud_or_mono
        aud_a = aud_a.reshape(-1, fram)
        part_min = np.max(np.abs(aud_a), axis=1)
        indices = np.where(part_min < 1e-3)[0]
        aud = np.delete(aud_a, indices, axis=0)
        aud = aud.reshape(-1)
        if len(aud) == 0:
            continue
        length = round(len(aud) / 2 + 0.4)
        power_spec = np.abs(scf.fft(aud)[:length]) ** 2
        power_spec_db = lib.power_to_db(power_spec)
        A_filter = lib.A_weighting(np.linspace(0, sr / 2, length))
        power_spec_A = lib.db_to_power(power_spec_db + A_filter)
        A_power = np.sqrt(np.sum(power_spec_A)) / len(aud)
        A_list.append(A_power)
    A_power = np.max(np.array(A_list))
    return A_power


def subfolder_loud_same(folder, ftype='.wav'):
    """
    将一个文件夹内的音频响度对齐，会修改原音频，注意做好备份
    :param folder: 文件夹路径，str
    :param ftype: 音频拓展名，默认'.wav'，记得写.,str
    :return: 无
    """
    all_music_path = dio.subfolder_filepath_read(folder, ftype)
    loud_list = []
    print('start read')
    for music_path in all_music_path:
        aud = lib.load(music_path, sr=sr, mono=False)[0]
        aud = audio_norm(aud)
        loud_list.append(A_weight_power(aud))
    min_loud = np.min(np.array(loud_list))
    print('get loud')
    for music_path in all_music_path:
        aud = lib.load(music_path, sr=sr, mono=False)[0]
        aud = audio_norm(aud)
        aud = aud / A_weight_power(aud) * min_loud
        write(music_path, 48000, aud.T)
    return all_music_path


def IACC(aud1, aud2, scal=[0,0], mode='full'):
    """
    计算两段音频的IACC
    :param aud1: 一段音频，单声道，array
    :param aud2: 令一段音频，单声道，array
    :param scal: IACC的范围，list(2,)
    :param mode: 相关计算的种类，默认’full‘
    :return: IACC值，位置点
    """
    ### 长度对齐
    length1 = len(aud1)
    length2 = len(aud2)
    if length1 >= length2:
        diff = length1 - length2
        diff1 = round(diff / 2)
        diff2 = diff - diff1
        aud2 = np.concatenate([np.zeros(diff1), aud2, np.zeros(diff2)])
    else:
        diff = length2 - length1
        diff1 = round(diff / 2)
        diff2 = diff - diff1
        aud1 = np.concatenate([np.zeros(diff1), aud1, np.zeros(diff2)])

    ### 两种相关计算，并剪裁
    if mode == 'same':
        IACF = scs.correlate(aud1, aud2, mode) / np.sqrt(sig_energy(aud1) * sig_energy(aud2))
    else:
        IACF = scs.correlate(aud1, aud2, mode) / np.sqrt(sig_energy(aud1) * sig_energy(aud2))

    ### 计算位置和最大值
    length = len(IACF)
    center_point = round(length / 2)
    if scal != [0,0]:
        IACF = IACF[scal[0] + center_point:scal[1] + center_point]
    length = len(IACF)
    center_point = round(length / 2)

    return np.max(IACF), np.argmax(IACF) - center_point


def FIACC(aud1, aud2, sr=48000, band=[0,0]):
    """
    计算FIACC函数，返回函数和x轴
    :param aud1: 一段音频，一维array
    :param aud2: 另一段音频，一维array
    :param sr: 采样率，默认48000
    :param band: 计算的频带，默认全频带
    :return:
    """
    af1 = scf.fft(aud1)
    af2 = scf.fft(aud2)
    f = np.linspace(0, sr/2, round(len(aud1)/2 + 0.9))
    fiacc = (np.real(af1 * af2.conj()) / np.abs(af1) / np.abs(af2))[:round(len(aud1)/2 + 0.9)]
    if band != [0,0]:
        f = f[round(band[0] / sr * len(aud1)) : round(band[1] / sr * len(aud1))]
        fiacc = fiacc[round(band[0] / sr * len(aud1)) : round(band[1] / sr * len(aud1))]
    return fiacc, f

def LP_filter(signal, freq, sr=48000, return_f=False):
    sig_len = len(signal)
    f = scs.firwin(sig_len*2, freq, fs=sr)
    signal = scs.convolve(f, signal, 'full')[sig_len:2*sig_len]
    if return_f:
        return signal, f
    else:
        return signal


def BP_filter(signal, freq_begin, freq_end, sr=48000, return_f=False):
    """
    FIR带通滤波，选用的滤波器点数很多，适合精确计算
    :param signal: 输入信号，一维array
    :param freq_begin: 通带，Hz
    :param freq_end: 阻带，Hz
    :param sr: 采样率
    :param return_f: 是否返回滤波器本身，默认否
    :return: 滤波后的信号，一维array
    """
    sig_len = len(signal)
    f = scs.firwin(sig_len*2, [freq_begin, freq_end], fs=sr, pass_zero=False)
    signal = scs.convolve(f, signal, 'full')[sig_len:2*sig_len]
    if return_f:
        return signal, f
    else:
        return signal


def part_mean(signal, win, pad=False):
    """
    给信号卷积矩形窗获得平滑后的曲线
    :param signal: 输入信号，一维array
    :param win: 窗长率，即窗长/信号总长
    :param pad: 是否在信号两侧进行延长
    :return: 返回平滑后的信号
    """
    ori = len(signal)
    win_len = round(len(signal) * win)
    win_func = np.ones(win_len) / win_len
    if pad:
        signal = np.pad(signal, (round(win_len / 2), round(win_len / 2)), mode='edge')
        signal = scs.convolve(win_func, signal, 'valid')[:ori]
    else:
        signal = scs.convolve(win_func, signal, 'full')[round(win_len / 2):round(win_len / 2 + len(signal))]
    return signal


def get_freq_weight(rir_db, band=[19, 21000], sr=48000):
    """
    根据输入的时域信号计算频域加权的权值向量，用于对频带进行对数加权计算
    :param rir_db: 音频信号，array，单声道
    :param band: 加权的频带，list，包含频带上下限
    :param sr: 采样率
    :return: 返回权值array和两个采样点的list
    """
    dpi = len(rir_db) / sr * 2
    hear_band_rir = rir_db[round(band[0]*dpi): round(band[1]*dpi)]
    length = len(hear_band_rir)
    f = np.linspace(band[0], band[1], length+1)
    f = np.log10(f)
    weight = scs.convolve(f, np.array([1,-1]), 'valid')
    weight = weight / np.max(weight)
    return weight, [round(band[0]*dpi), round(band[1]*dpi)]


def is_mono_audio(audio):
    """
    判断输入是否为audio
    :param audio: array格式的audio
    :return:
    """
    if audio.ndim == 1:  # ndim 属性表示音频的维度，单声道音频维度为 1
        return True
    else:
        return False


def multi_ch_convolve(aud1, aud2, mode='full', scale='head'):
    """
    多通道卷积
    :param aud1: 原音频，array，要求通道数为一或等于aud2
    :param aud2: 卷积核，array，要求通道数为一或等于aud1
    :param mode: 卷积形式，默认full，str
    :param scale: 若选择’full‘，卷积核若为ir类选head，默认head，str
    :return: 卷积后信号
    """
    if np.ndim(aud1) == 1:  # 单声道音频转成二维矩阵，方便遍历
        aud1 = aud1.reshape(1, -1)
    if np.ndim(aud2) == 1:  # 单声道音频转成二维矩阵，方便遍历
        aud2 = aud2.reshape(1, -1)
    if scale == 'head':
        ac = [0, len(aud1[0])]
    else:
        ac = [round(len(aud2[0]) / 2), round(len(aud2[0]) / 2) + len(aud1[0])]
    # print([len(aud1), len(aud2)])
    # print([len(aud1) == 1 , len(aud2) != 1, len(aud1) != 1 , len(aud2) == 1])
    audio_list = []
    if len(aud1) == len(aud2):
        for a1, a2 in zip(aud1, aud2):
            audio_list.append(scs.convolve(a1, a2, mode)[ac[0]:ac[1]])
    elif (len(aud1) == 1) & (len(aud2) != 1):
        aud1 = aud1.reshape(-1)
        for a2 in aud2:
            audio_list.append(scs.convolve(aud1, a2, mode)[ac[0]:ac[1]])
    elif (len(aud1) != 1) & (len(aud2) == 1):
        aud2 = aud2.reshape(-1)
        for a1 in aud1:
            audio_list.append(scs.convolve(a1, aud2, mode)[ac[0]:ac[1]])
    audio_out = np.array(audio_list)
    return audio_out


def MAX(sig):
    """
    寻找信号的绝对值最大值
    :param sig: 输入信号array
    :return: 最大值
    """
    return np.max(np.abs(sig))


def sp_to_filter(sp, even_or_odd):
    """
    将一段频响转换成FIR滤波器
    :param sp: 输入频响
    :param even_or_odd: ’even‘或者’odd‘，滤波器点数为奇数还是偶数
    :return: FIR滤波器
    """
    if even_or_odd == 'even':
        sp = np.concatenate([sp, np.conj(sp[-2:0:-1])])
    elif even_or_odd == 'odd':
        sp = np.concatenate([sp, np.conj(sp[-1:0:-1])])
    else:
        return
    filterr = np.real(scf.ifftshift(scf.ifft(sp)))
    return filterr


def FIR_filter_band_change(f, gain):
    """
    将一个滤波器转换为EQ的样子
    :param f: 滤波器函数，一维array
    :param gain: 滤波器的增益，dB形式
    :return: 新的滤波器
    """
    gain = 10 ** (gain / 20)
    weight = gain - 1
    p_point = np.argmax(f)
    f_n = f * weight
    f_n[p_point] = f_n[p_point] + 1
    return f_n / sig_energy(f_n, True)



def list_array(list_list):
    max_len = 0
    for list_s in list_list:
        if len(list_s) > max_len:
            max_len = len(list_s)

    larray = []
    for list_s in list_list:
        list_vec = np.array(list_s)
        list_vec_len = np.concatenate([list_vec, np.zeros(max_len - len(list_vec))])
        larray.append(list_vec_len)
    return np.array(larray)


def list_array2(list_list):
    max_len = 0
    for list_s in list_list:
        if len(list_s[0]) > max_len:
            max_len = len(list_s[0])

    larray = []
    for list_s in list_list:
        list_vec = np.array(list_s)
        pad_array = np.zeros([len(list_s), max_len - len(list_vec[0])])
        list_vec_len = np.concatenate([list_vec, pad_array], axis=1)
        larray.append(list_vec_len)
    return np.array(larray)



def array_add(array_list):

    if np.ndim(array_list[0]) == 1:
        list_s = list_array(array_list)
        return np.sum(list_s, axis=0)
    else:
        array_add_vec = []
        ch_len = len(array_list[0])
        for i in range(ch_len):
            list_s = []
            for array_s in array_list:
                list_s.append(array_s[i])
            list_s = list_array(list_s)
            array_add_vec.append(np.sum(list_s, axis=0))
        return list_array(array_add_vec)



if __name__ == "__main__":
    list1 = [[1,1,1], [1,1,1,1],[1,1], [1,1,1,1,1]]
    array1 = array_add(list1)
    list2 = [np.ones([4,2]), np.ones([4,4]), np.ones([4,5]), np.ones([4,3])]
    array2 = list_array2(list2)
    # s = np.ones(240000)
    # name = 'sp_L6'
    # _, f1 = LP_filter(s, 300, return_f=True)
    # f1 = FIR_filter_band_change(f1, 6)
    # dio.output(f1, name + '.wav', 'D:\\Lizy\\Python\\RIR_tools\\audio_full')
    # plt.loglog(np.abs(scf.fft(f1)), label = name)
    #
    # name = 'sp_L-6'
    # _, f1 = LP_filter(s, 300, return_f=True)
    # f1 = FIR_filter_band_change(f1, -6)
    # dio.output(f1, name + '.wav', 'D:\\Lizy\\Python\\RIR_tools\\audio_full')
    # plt.loglog(np.abs(scf.fft(f1)), label = name)
    #
    # name = 'sp_ML6'
    # _, f1 = BP_filter(s, 300, 2000, return_f=True)
    # f1 = FIR_filter_band_change(f1, 6)
    # dio.output(f1, name + '.wav', 'D:\\Lizy\\Python\\RIR_tools\\audio_full')
    # plt.loglog(np.abs(scf.fft(f1)), label = name)
    #
    # name = 'sp_ML-6'
    # _, f1 = BP_filter(s, 300, 2000, return_f=True)
    # f1 = FIR_filter_band_change(f1, -6)
    # dio.output(f1, name + '.wav', 'D:\\Lizy\\Python\\RIR_tools\\audio_full')
    # plt.loglog(np.abs(scf.fft(f1)), label = name)
    #
    # name = 'sp_MH6'
    # _, f1 = BP_filter(s, 2000, 5000, return_f=True)
    # f1 = FIR_filter_band_change(f1, 6)
    # dio.output(f1, name + '.wav', 'D:\\Lizy\\Python\\RIR_tools\\audio_full')
    # plt.loglog(np.abs(scf.fft(f1)), label = name)
    #
    # name = 'sp_MH-6'
    # _, f1 = BP_filter(s, 2000, 5000, return_f=True)
    # f1 = FIR_filter_band_change(f1, -6)
    # dio.output(f1, name + '.wav', 'D:\\Lizy\\Python\\RIR_tools\\audio_full')
    # plt.loglog(np.abs(scf.fft(f1)), label = name)
    #
    # name = 'sp_H6'
    # _, f1 = LP_filter(s, 5000, return_f=True)
    # f1 = FIR_filter_band_change(f1, -6)
    # dio.output(f1, name + '.wav', 'D:\\Lizy\\Python\\RIR_tools\\audio_full')
    # plt.loglog(np.abs(scf.fft(f1)), label = name)
    #
    # name = 'sp_H-6'
    # _, f1 = LP_filter(s, 5000, return_f=True)
    # f1 = FIR_filter_band_change(f1, 6)
    # dio.output(f1, name + '.wav', 'D:\\Lizy\\Python\\RIR_tools\\audio_full')
    # plt.loglog(np.abs(scf.fft(f1)), label = name)
    #
    # plt.legend()
    # plt.grid()
    # audio_f = 'D:\\Lizy\\Python\\RIR_tools\\audio_full\\audio\\'
    # audio_m, _, audio_p = dio.folder_file_read(audio_f, return_type='all')
    # for i in range(len(audio_m)):
    #     if i % 6 == 0:
    #         plt.figure()
    #     name = audio_m[i]
    #     fil = dio.load(audio_p[i])
    #     energy = sig_energy(fil)
    #     f_len = len(fil)
    #     sp_len = round(f_len / 2) + 1
    #     f = np.linspace(0, 24000, sp_len)
    #     f_log = np.abs(scf.fft(fil))[:sp_len]
    #     plt.loglog(f, f_log, label = name + '_{}'.format(str(np.round(energy, 2))))
    #     if i % 6 == 5:
    #         plt.grid()
    #         plt.legend()
    #
    #
    # audio_f = 'D:\\Lizy\\Python\\RIR_tools\\audio_full\\audio\\'
    # _, audio_m, audio_p = dio.folder_file_read(audio_f, return_type='all')
    # for name, path in zip(audio_m, audio_p):
    #     au = dio.load(path)
    #     dio.output(au / sig_energy(au, True), name, audio_f)