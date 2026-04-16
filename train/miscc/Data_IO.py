import pickle
import librosa as lib
import numpy as np
import pandas as pd
import scipy.fft as scf
import scipy.signal as scs
import librosa.display as libd
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import json
import re
import soundfile as sf
import scipy.io.wavfile as sciw
import pickle

sr = 48000

def load_pickle(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data


def save_pickle(data, file_path, info_show=True):
    if not any(sub in file_path for sub in ['/SRIR/','/SRIR2/', '/ER_Nd/', '/Ely_Reflc/']):
        with open(file_path, 'wb') as f:
            pickle.dump(data, f)
        if info_show:
            print('pickle保存路径{}'.format(file_path))
    else:
        print('路径有误，往原始数据里存东西了')
    return None


def create_folder(folder_path, folder_name):
    """
    创建文件夹
    :param folder_path: 创建文件夹的路径, str
    :param folder_name: 文件夹的名字, str
    :return:
    """
    folder_to_create = os.path.join(folder_path, folder_name)
    try:
        os.mkdir(folder_to_create)
        print(f"文件夹 '{folder_name}' 已成功创建在目录 '{folder_path}' 中。")
    except FileExistsError:
        print(f"文件夹 '{folder_name}' 已经存在于目录 '{folder_path}' 中。")
    except Exception as e:
        print(f"创建文件夹时发生错误：{str(e)}")
    return os.path.join(folder_path, folder_name)


def folder_file_read(directory, extension='.wav', return_type='file'):
    """
    读取文件夹内所有包含特定拓展名的文件
    :param directory: 文件夹的路径, str
    :param extension: 文件拓展名, str, 需要加.
    :param return_type: 返回数据的种类，不含拓展名的文件’name‘，含拓展名’file‘，路径’path‘
    :return: 符合要求的list
    """
    names = []
    file_names = []
    path = []
    for file in os.listdir(directory):
        if os.path.isfile(os.path.join(directory, file)):
            file_name = os.path.splitext(file)  # 获取不包含拓展名的文件名
            if file_name[1] != extension:
                continue
            names.append(file_name[0])
            file_names.append(file_name[0] + file_name[1])
            path.append(os.path.join(directory, file_name[0] + file_name[1]))
    if return_type == 'name':
        return names
    elif return_type == 'file':
        return file_names
    elif return_type == 'all':
        return names, file_names, path
    else:
        return path


def subfolder_filepath_read(folder, extension='.wav', exclude_keyword='', include_keyword=''):
    """
    读取一个文件夹及其所有子文件夹的特定类型的文件的路径
    :param folder: 文件夹的路径, str
    :param extension: 文件拓展名, str, 需要加.
    :param exclude_keyword: 需要排除掉的关键词或关键词列表，str or list
    :param include_keyword:  必须包含的的关键词或关键词列表，str or list
    :return: 路径列表，list
    """
    wave_files_ori = []
    # 遍历文件夹中的所有文件和子文件夹
    for root, dirs, files in os.walk(folder):
        # 遍历当前文件夹中的所有文件
        for file in files:
            if file.endswith(extension):  # 只处理扩展名为 .wav 的文件
                wave_files_ori.append(os.path.join(root, file))
        # 遍历当前文件夹中的所有子文件夹
        for dir in dirs:
            wave_files_ori.extend(subfolder_filepath_read(os.path.join(root, dir), extension))

    # 去除重复的path
    wave_files = [x for i, x in enumerate(wave_files_ori) if x not in wave_files_ori[:i]]

    #
    if exclude_keyword != '':
        if type(exclude_keyword) == type('key'):
            wave_files = [string for string in wave_files if exclude_keyword not in string]
        else:
            for word in exclude_keyword:
                wave_files = [string for string in wave_files if word not in string]
    if include_keyword != '':
        if type(include_keyword) == type('key'):
            wave_files = [string for string in wave_files if include_keyword in string]
        else:
            wave_files_part = []
            for word in include_keyword:
                wave_files_part = wave_files_part + [string for string in wave_files if word in string]
            wave_files = wave_files_part
    return wave_files




def spl(second):
    """
    根据文件data_IO.py中预设的采样率值（sr一般为48000），将输入的时间（单位：秒）转化为采样点数输出

    spl：时间转采样点
    sec：采样点转时间

    :param second: 时间/秒
    :return: 采样点数
    """
    return round(second*sr)


def sec(sample):
    """
    根据文件data_IO.py中预设的采样率值（sr一般为48000），将输入的采样点数转化为时间（单位：秒）输出

    sec：采样点转时间
    spl：时间转采样点

    :param sample: 采样点数
    :return: 时间/秒
    """
    return sample/sr


def dictionary_save(dic, file_name, file_path = ''):
    """
    将字典对象保存为.pickle文件

    :param dic: 待保存的字典
    :param file_name: 字典文件名（不用包含.pickle）
    :param file_path: 字典文件的文件夹路径，默认为空
    :return:
    """
    with open(file_path + file_name + '.pickle', 'wb') as ff:
        pickle.dump(dic, ff)
    return

# 保存字典
def dictionary_read(file_name, file_path = ''):
    """
    读取.pickle文件到字典对象

    :param file_name: 字典文件名（不用包含.pickle）
    :param file_path: 字典文件的文件夹路径，默认为空
    :return: .pickle文件包含的字典对象
    """
    with open(file_path + file_name + '.pickle', 'rb') as ff:
        dic = pickle.load(ff)
    return dic


def dictionary2xlsx(dic, dimension, file_name, file_path = ''):
    """
    将字典导出到Excel，注意字典的内容只能为一维的列表或者向量对象，并且维度要一样

    :param dic: 需要导出的字典对象
    :param dimension: 字典内容的维度
    :param file_name: Excel文件名（不用包含.xlsx）
    :param file_path: Excel文件的文件夹路径，默认为空
    :return:
    """
    df = pd.DataFrame.from_dict(dic, orient='index', columns=['value']*dimension)
    df.to_excel(os.path.join(file_path, file_name+'.xlsx'), index=True)
    return


def T10(signal):
    """
    计算房间脉冲响应的T60值

    参数：
    signal: ndarray，输入的房间脉冲响应信号
    fs: int，信号的采样率

    返回：
    t60: float，房间的T60值（以秒为单位）
    """
    initial_energy = np.sum(signal ** 2) # 房间脉冲响应的初始能量
    signal_length = len(signal) # 信号的长度
    threshold_energy = (1-0.1) * initial_energy # T60为信号衰减到初始能量的0.001倍所需的时间
    accumulated_energy = 0 # 计算T60值
    t10 = 0
    for i in range(signal_length):
        accumulated_energy += signal[i] ** 2
        if accumulated_energy >= threshold_energy:
            t10 = i
            break
    return t10


def peak_detect(rir):
    """
    分离峰值和rir的函数
    :param rir:
    :return:
    """
    x = np.linspace(-24, 24, 96)
    x = np.sin(x * np.pi / 12) / x
    x = x / energy(x) / 96
    rir = np.abs(rir)
    rir_base = scs.convolve(rir, np.ones(240)/240, 'same') * 0.7
    rir_p = scs.convolve(rir, x, 'same')
    peako = (np.sign(rir_p - rir_base) + 1) / 2

    signal_length = len(rir) # 信号的长度
    for i in range(signal_length):
        if abs(rir[i]) > 0.1:
            st_point = i
            break
    end_point = T10(rir)
    peak = np.zeros(signal_length)
    peak[st_point: round(end_point * 48000 + st_point)] = peako[st_point: round(end_point * 48000 + st_point)]
    return peak, rir_base, rir_p


def energy(signal):
    """
    计算信号各点平均能量
    :param signal: 信号，array
    :return: 平均能量，float
    """
    return np.sqrt(np.mean(np.square(np.abs(signal))))


def RIR_show(num, show = True):
    """
    这函数需要修改，本来是从特定的文件夹里去除一个rir然后展示信息，但是文件夹删了
    :param num:
    :param show:
    :return:
    """
    print(num)
    path = 'D:/Lizy/Python/RIR_tools/open AIR/' + str(num).zfill(4)
    rir_dic = dictionary_read(path)
    rir = rir_dic['rir']
    if show:
        len_rir = len(rir)
        dpi = len_rir / 24000

        peak, rir_base, rir_p = peak_detect(rir)

        fig = plt.figure(figsize=(14, 10), dpi=300)
        fig.suptitle(rir_dic['path'][42:-4] + '\n' + str(rir_dic['ID']).zfill(4))

        ax1 = fig.add_subplot(2,2,1)
        t = np.linspace(0, (len_rir-1)/48000, len_rir)
        ax1.plot(t[:round(48000*rir_dic['T50'])], rir[:round(48000*rir_dic['T50'])], linewidth=0.3)
        ax1.plot(t[:round(48000 * rir_dic['T50'])],
            rir[:round(48000 * rir_dic['T50'])] * rir_dic['peak']['peak_loc'][:round(48000 * rir_dic['T50'])],
            linewidth=0.15)
        ax1.set_title('RIR wave    T30:{:.{}f}s    T60:{:.{}f}s    Time center:{:.{}f}s'.format(rir_dic['T30'], 3,
                                                                 rir_dic['T60'], 3, rir_dic['time_center'], 3))
        ax1.set_xlabel('Time/t')
        ax1.set_ylabel('Magnitude')
        ax1.set_ylim(-1, 1)
        ax1.grid()

        ax2 = fig.add_subplot(2,2,2)
        f = np.linspace(0, 24000, round(len_rir/2 +1))
        rir_fft = np.abs(scf.fft(rir))[0:round(len_rir/2 +1)]
        rir_fft = rir_fft/np.max(rir_fft)
        rir_fft_tend = scs.convolve(rir_fft, np.ones(round(len_rir/200))/round(len_rir/200), 'same')
        rir_fft_tend = rir_fft_tend / energy(rir_fft_tend) * energy(rir_fft)
        ax2.plot(f, rir_fft, linewidth=0.3)
        ax2.plot(f, rir_fft_tend)
        ax2.set_title('RIR FFT    Freq center:{}Hz     Var:{:.{}f}'.format(round(rir_dic['freq_center']),
                                                                        rir_dic['var'], 3))
        ax2.set_xlabel('Frequency/Hz')
        ax2.set_ylabel('Magnitude')
        ax2.grid()

        ax3 = fig.add_subplot(2,2,4)
        f_db = f[round(10*dpi):round(20000*dpi)]
        rir_fft_db = rir_fft[round(10*dpi):round(20000*dpi)]
        rir_fft_db = rir_fft_db/np.max(rir_fft_db)
        rir_fft_db_tend = scs.convolve(rir_fft_db, np.ones(round(len_rir/200))/round(len_rir/200), 'same')
        rir_fft_db_tend = rir_fft_db_tend - np.mean(rir_fft_db_tend) + np.mean(rir_fft_db)
        ax3.loglog(f_db, rir_fft_db, linewidth=0.3)
        ax3.loglog(f_db, rir_fft_db_tend)
        ax3.set_title('RIR FFT    Freq center dB:{}Hz     energy:{:.{}f}J'.format(
            round(rir_dic['freq_db_center']), rir_dic['energy'], 2))
        ax3.set_xlabel('Frequency/Hz')
        ax3.set_ylabel('Magnitude')
        ax3.set_ylim(1e-4, 1)
        ax3.grid()

        ax4 = fig.add_subplot(2,2,3)
        n_fft = 256  # FFT 窗口大小
        hop_length = 2  # 帧移大小
        stft = lib.stft(rir, n_fft=n_fft, hop_length=hop_length)
        # 计算幅度谱的对数值
        amplitude = np.abs(stft)
        amplitude = amplitude / np.max(amplitude)
        log_amplitude = lib.amplitude_to_db(amplitude)
        # 绘制 STFT 的幅度谱图，使用对数坐标
        libd.specshow(log_amplitude, sr=48000, hop_length=hop_length, x_axis='time')
        #ax4.figure.colorbar(img, ax=ax4, format='%+2.0f dB')  # 显示颜色条
        ax4.set_title('Log Spectrogram of    Peak num:{}    Peak energy:{:.{}f}'.format(
            rir_dic['peak_num'], rir_dic['peak_energy_rate'], 3))
        ax4.set_xlabel('Time (s)')
        ax4.set_ylabel('Frequency (Hz)')

        plt.savefig('D:/Lizy/Python/RIR_tools/open AIR/Figure/{}.jpg'.format(str(rir_dic['ID']).zfill(4)))
        plt.clf()
        return


def RIR(num, dic = True):
    path = 'D:/Lizy/Python/RIR_tools/open AIR/' + str(num).zfill(4)
    rir_dic = dictionary_read(path)
    rir = rir_dic['rir']
    if dic:
        return rir_dic
    else:
        return rir


def json2xlsx(path_in, file_out, path_out=''):
    """
    将mushra测试软件生成的测试结果的json文件转换成Excel
    :param path_in: json文件路径
    :param file_out: 输出的Excel文件名，不用包含拓展名
    :param path_out: 输出的Excel文件路径
    :return: 输出一个包含json信息的dic
    """
    with open(path_in, 'r', encoding='utf-8') as json_file:
        data = json.load(json_file)
        data = data['results']
        del data['test1']
        del data['test2']
        out_dic = {}
        key1_list = list(data.keys())
        for key1 in key1_list:
            single_score = data[key1]
            key2_list = list(single_score.keys())
            for key2 in key2_list:
                score = single_score[key2]
                delimiter_pattern = r'[_:]'
                key2_split = re.split(delimiter_pattern, key2)
                key2_split.append('score')
                key2_split.append(score)
                dim = len(key2_split)
                out_dic['{}_{}'.format(key1, key2)] = key2_split
    dictionary2xlsx(out_dic, dim, file_out, path_out)
    return out_dic


def load(path, return_sr=False, resample=48000):
    """
    加载音频
    :param path: 音频文件路径
    :param return_sr: 是否返回音频文件的真实采样率
    :param resample: 读取的音频文件采样率，默认48000，不满足则重采样
    :return: 音频array，或者array，int采样率
    """
    wav_head = sf.SoundFile(path)
    sr = wav_head.samplerate
    audio = lib.load(path, sr=sr, mono=False)[0]
    if resample != 0:
        if sr != resample:
            audio = lib.resample(audio, orig_sr=sr, target_sr=resample)
            sr = resample
    if return_sr:
        return audio, sr
    else:
        return audio


def output(aud, name, f_path='./', sr=48000, show_info=True):
    """
    音频输出，可以自动调整矩阵维度，是否需要转置
    :param aud: 音频array
    :param name: 文件名，包含拓展名
    :param f_path: 文件路径，默认‘./’
    :param sr: 默认48000
    :return:
    """
    path = os.path.join(f_path, name)
    aud = np.array(aud)
    if np.ndim(aud) == 1:
        sciw.write(path, sr, aud)
        if show_info:
            print('{}: 1 channel, {} second'.format(name, np.round(aud.shape[0] / sr, 3)))

    else:
        if aud.shape[0] >= 64:
            aud = aud.T
        sciw.write(path, sr, aud.T)
        if show_info:
            print('{}: {} channel, {} second'.format(name, aud.shape[0], np.round(aud.shape[1] / sr, 3)))
    return


def signal_show(signal, freq=True, signal_labels=[], figtyp=['line', 'line'], sr=48000):
    """
    可视化输入信号
    :param signal: 可视化的信号
    :param freq: 是否显示频谱
    :param signal_labels: 手动设置显示信号的label，当signal为多声道时可以使用，[str，str，……]
    :param figtyp: 图像的x，y轴为线性还是对数，[str, str], line or log
    :param sr: 采样率
    :return:
    """
    if np.ndim(signal) == 1:
        signal = signal.reshape(1, -1)
    if signal_labels == []:
        signal_labels = np.arange(0, signal.shape[0], 1).tolist()
    t = np.arange(0, len(signal[0]), 1) / sr
    fig = plt.figure()
    if freq:
        ax1 = fig.add_subplot(2, 1, 1)
    else:
        ax1 = fig.add_subplot(1, 1, 1)
    for ch, label in zip(signal, signal_labels):
        ax1.plot(t, ch, label=label)
    ax1.grid()
    ax1.set_xlim([0, len(signal[0])/sr])
    ax1.legend()

    if freq:
        ax2 = fig.add_subplot(2, 1, 2)
        freq_len = round(len(signal[0]) / 2 + 0.1)
        f = np.linspace(0, sr / 2, freq_len)
        freq = np.abs(scf.fft(signal, axis=1)[:, :freq_len])
        if figtyp[1] == 'log':
            freq = lib.amplitude_to_db(freq, top_db=120)
        for ch, label in zip(freq, signal_labels):
            ax2.plot(f, ch, label=label)
        if figtyp[0] == 'log':
            ax2.set_xlim([20, 20000])
            ax2.set_xscale('log')
        ax2.grid()
        ax2.legend()
    plt.show()
    return









if __name__ == "__main__":
    d = 'D:/Lizy/Python/RIR_tools/clarity_evaluation/T30_test/231109李知禹_T30_DRR1.8_Test'
    json2xlsx(d + '.json', d)

