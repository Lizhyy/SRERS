import sys  # 导入系统模块，用于追加项目路径
import os  # 导入操作系统模块，用于读取环境变量等
from pathlib import Path  # 导入 Path，用于跨平台处理 filter_bank.npy 路径
from typing import Optional, Tuple, Union  # 导入类型标注；Union 当前保留以兼容后续扩展

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # 获取 SRERS 项目根目录
if str(PROJECT_ROOT) not in sys.path:  # 避免重复加入 Python 搜索路径
    sys.path.insert(0, str(PROJECT_ROOT))  # 将项目根目录加入搜索路径，支持发布版相对导入

import numpy as np  # 导入 NumPy，用于兼容原单样本接口和加载 filter_bank.npy
import torch  # 导入 PyTorch，用于 GPU/CPU 张量并行解码
import torch.nn.functional as F  # 导入 torch functional；当前文件暂未使用，保留原导入结构

try:  # 尝试读取项目配置，保证在 SRERS 工程环境中使用真实 cfg
    from CConfig import config  # 导入项目配置类
    cfg = config()  # 实例化全局配置对象
except Exception:  # 如果脱离工程环境运行，允许 decoder 以默认行为继续工作
    cfg = None  # cfg 置空，后续通过 _runtime_flags 给出默认开关


# --------------------------------------------------------
# 全局缓存：filter_bank、filter FFT 和随机数生成器
# --------------------------------------------------------
_FILTER_BANK_CACHE = {}  # 按 device/dtype 缓存 filter_bank，避免每次解码重复从磁盘加载
_FILTER_FFT_CACHE = {}  # 按 n_fft/device/dtype 缓存滤波器频域结果，避免重复 FFT
_RNG_CACHE = {}  # 按 device 缓存 torch.Generator，保证噪声生成器只初始化一次
_EPS = 1e-12  # 数值稳定项，避免除零和 sqrt/log 等数值问题


def _candidate_filter_paths():
    '''
    返回可能存在 filter_bank.npy 的路径列表。

    设计目的：
    1. 兼容当前文件被放在 train/para_dataset、train/miscc 或其他相邻目录的情况；
    2. 保留原工程中的绝对路径兜底；
    3. 只负责给出候选路径，不做文件读取，实际读取由 _get_filter_bank 完成。
    '''
    here = Path(__file__).resolve()  # 获取当前 Python 文件的绝对路径，便于构造相对路径
    return [  # 返回按优先级排列的候选路径列表
        here.parent / 'filter_bank.npy',  # 候选 1：当前文件同目录
        here.parent.parent / 'filter_bank.npy',  # 候选 2：当前文件上一级目录
        here.parent.parent / 'miscc' / 'filter_bank.npy',  # 候选 3：上一级目录下的 miscc 目录
        PROJECT_ROOT / 'train' / 'filter_bank.npy',  # 候选 4：发布版 train 目录
        PROJECT_ROOT / 'train' / 'miscc' / 'filter_bank.npy',  # 候选 5：发布版 train/miscc 目录
        Path.cwd() / 'filter_bank.npy',  # 候选 6：当前工作目录兜底
    ]


def _get_filter_bank(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    '''
    获取指定 device 和 dtype 上的 filter_bank 张量。

    参数：
        device: 目标设备，例如 cuda:0 或 cpu。
        dtype: 目标数据类型，通常为 torch.float32。

    返回：
        形状为 [n_band, filter_len] 的滤波器组张量。

    注意：
        该函数使用全局缓存，第一次调用时从磁盘加载，之后直接复用 GPU/CPU 上的张量。
    '''
    key = (str(device), dtype)  # 用设备字符串和 dtype 共同作为缓存键
    if key not in _FILTER_BANK_CACHE:  # 如果当前设备和精度下还没有缓存，则从磁盘加载
        filter_path = None  # 初始化实际命中的 filter_bank.npy 路径
        for p in _candidate_filter_paths():  # 逐个检查候选路径
            if p.exists():  # 找到第一个存在的 filter_bank.npy
                filter_path = p  # 记录命中的路径
                break  # 停止继续搜索
        if filter_path is None:  # 如果所有候选路径都不存在，则说明环境缺少必要文件
            raise FileNotFoundError('Cannot find filter_bank.npy. Please put it under SRERS/train or SRERS/train/miscc.')  # 抛出明确错误
        filt = np.load(str(filter_path))  # 用 NumPy 读取原始 filter_bank.npy，保持与原工程格式兼容
        _FILTER_BANK_CACHE[key] = torch.as_tensor(filt, device=device, dtype=dtype)  # 转为目标设备张量并缓存
    return _FILTER_BANK_CACHE[key]  # 返回缓存后的滤波器组张量


def _get_filter_fft(n_fft: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    '''
    获取指定 FFT 长度下的 filter_bank 频域表示。

    参数：
        n_fft: 频域卷积使用的 FFT 点数。
        device: 目标设备。
        dtype: 目标数据类型。

    返回：
        形状为 [n_band, n_fft//2+1] 的 rFFT 结果。

    说明：
        late reverb 的带限噪声通过频域卷积实现，因此滤波器 FFT 是高频调用对象，必须缓存。
    '''
    key = (n_fft, str(device), dtype)  # 将 FFT 长度、设备和精度作为缓存键
    if key not in _FILTER_FFT_CACHE:  # 如果该组合还没有频域滤波器缓存
        filt = _get_filter_bank(device, dtype)  # 先获取时域滤波器组
        _FILTER_FFT_CACHE[key] = torch.fft.rfft(filt, n=n_fft, dim=-1)  # 沿时间维做 rFFT 并缓存
    return _FILTER_FFT_CACHE[key]  # 返回缓存后的滤波器频域表示


def _get_rng(device: torch.device) -> torch.Generator:
    '''
    获取指定 device 上的随机数生成器。

    参数：
        device: 目标设备。

    返回：
        torch.Generator 对象。

    说明：
        原 NumPy 版本使用固定 seed=2026 的随机生成器；这里用 torch.Generator 对齐这一设计。
    '''
    key = str(device)  # 用设备字符串作为随机数生成器缓存键
    if key not in _RNG_CACHE:  # 如果当前设备还没有随机数生成器
        gen = torch.Generator(device=device)  # 在目标设备上创建随机数生成器
        gen.manual_seed(2026)  # 设置固定随机种子，保持推理过程可复现
        _RNG_CACHE[key] = gen  # 将生成器放入缓存
    return _RNG_CACHE[key]  # 返回缓存后的随机数生成器


def _as_tensor(x, device: Optional[torch.device] = None, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    '''
    将输入统一转换为目标 device/dtype 上的 torch.Tensor。

    参数：
        x: 输入数据，可以是 torch.Tensor、NumPy 数组或列表。
        device: 目标设备；若为 None，则优先沿用输入张量设备，否则自动选择 cuda/cpu。
        dtype: 目标数据类型。

    返回：
        转换后的 torch.Tensor。

    说明：
        该函数是兼容接口的关键，使 SRERS_decoder 既能接受原 NumPy 输入，也能接受推理阶段 GPU 张量。
    '''
    if isinstance(x, torch.Tensor):  # 如果输入本身已经是张量
        out = x.detach() if not x.requires_grad else x  # 推理时通常不需要梯度；若仍带梯度则保留原图
        if device is None:  # 如果调用方没有指定目标设备
            device = out.device  # 默认沿用输入张量所在设备
        return out.to(device=device, dtype=dtype, non_blocking=True)  # 转为目标设备和精度，GPU 间复制尽量异步
    if device is None:  # 如果输入不是张量且没有指定设备
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 优先使用 GPU，否则使用 CPU
    return torch.as_tensor(x, device=device, dtype=dtype)  # 将 NumPy/列表数据直接构造成目标设备张量


def _runtime_flags(io_rir_er_type: Optional[str] = None, with_lor: Optional[bool] = None) -> Tuple[str, bool]:
    '''
    读取运行时与 LoR 相关的配置开关。

    参数：
        io_rir_er_type: 外部显式指定的 IO_RIR_ER_type；若为 None，则从 cfg 读取。
        with_lor: 外部显式指定是否叠加 LoR；若为 None，则从 cfg 读取。

    返回：
        io_type: cfg.INIT.IO_RIR_ER_type 或外部传入值，缺省为 'ER'。
        use_lor: cfg.TEST.with_LoR 或外部传入值，缺省为 True。

    说明：
        当 io_type == 'RIR' 或 use_lor == False 时，解码阶段会将 LoR 置零，
        与原 SRERS_decoder 的逻辑保持一致。
    '''
    if cfg is None:  # 如果没有成功加载工程配置
        cfg_io_type = 'ER'  # 使用默认输入类型
        cfg_with_lor = True  # 默认启用 LoR
    else:  # 如果处于完整工程环境
        cfg_io_type = getattr(getattr(cfg, 'INIT', object()), 'IO_RIR_ER_type', 'ER')  # 安全读取 INIT.IO_RIR_ER_type
        cfg_with_lor = getattr(getattr(cfg, 'TEST', object()), 'with_LoR', True)  # 安全读取 TEST.with_LoR
    io_type = cfg_io_type if io_rir_er_type is None else io_rir_er_type  # 外部参数优先
    use_lor = cfg_with_lor if with_lor is None else with_lor  # 外部参数优先
    return io_type, bool(use_lor)  # 返回字符串开关和布尔开关


def _to_batch_inputs(SRIR_ER, para, reverb_ev, LoR, dtype: torch.dtype = torch.float32, device: Optional[torch.device] = None):
    '''
    将单样本输入或 batch 输入统一整理成 batch 形状。

    输入兼容：
        SRIR_ER: [C, T_er] 或 [B, C, T_er]
        para: [3] 或 [B, 3]
        reverb_ev: [C, n_band, T_ev] 或 [B, C, n_band, T_ev]
        LoR: [C, T_lor] 或 [B, C, T_lor]

    返回：
        SRIR_ER, para, reverb_ev, LoR: 统一后的 batch 张量；
        single_input: 是否原本为单样本输入，用于输出时恢复单样本格式。
    '''
    if device is None:  # 如果调用方没有显式指定目标设备
        for x in (SRIR_ER, para, reverb_ev, LoR):  # 遍历四类输入
            if isinstance(x, torch.Tensor):  # 只要有一个输入是张量
                device = x.device  # 就沿用该张量所在设备
                break  # 推断到设备后停止遍历
    if device is None:  # 如果所有输入都不是张量且没有指定设备
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 自动选择 GPU 或 CPU

    SRIR_ER = _as_tensor(SRIR_ER, device, dtype)  # 将 ER 波形转为目标设备张量
    para = _as_tensor(para, device, dtype)  # 将参数向量转为目标设备张量
    reverb_ev = _as_tensor(reverb_ev, device, dtype)  # 将 late 能量包络转为目标设备张量
    LoR = _as_tensor(LoR, device, dtype)  # 将 LoR 波形转为目标设备张量

    single_input = False  # 默认认为输入是 batch
    if SRIR_ER.ndim == 2:  # 单样本 ER 形状为 [C, T]
        SRIR_ER = SRIR_ER.unsqueeze(0)  # 添加 batch 维，变为 [1, C, T]
        single_input = True  # 记录输入原本是单样本
    if para.ndim == 1:  # 单样本参数形状为 [3]
        para = para.unsqueeze(0)  # 添加 batch 维，变为 [1, 3]
    if reverb_ev.ndim == 3:  # 单样本包络形状为 [C, n_band, T_ev]
        reverb_ev = reverb_ev.unsqueeze(0)  # 添加 batch 维，变为 [1, C, n_band, T_ev]
    if LoR.ndim == 2:  # 单样本 LoR 形状为 [C, T_lor]
        LoR = LoR.unsqueeze(0)  # 添加 batch 维，变为 [1, C, T_lor]

    return SRIR_ER, para, reverb_ev, LoR, single_input  # 返回整理后的 batch 输入和单样本标记


@torch.no_grad()  # 解码阶段不需要梯度，关闭 autograd 可以节省显存和时间
def _band_pass_noise_batch(num_streams: int,
                           sig_len: int,
                           t60: torch.Tensor,
                           device: torch.device,
                           dtype: torch.dtype) -> torch.Tensor:
    '''
    批量生成带限噪声。

    参数：
        num_streams: 独立噪声流数量，通常等于 batch_size * channel_num。
        sig_len: 输出噪声的统一 padding 长度，一般为 batch 内最大 T60。
        t60: 每条噪声流自己的真实有效长度，形状为 [num_streams]。
        device: 目标设备。
        dtype: 目标数据类型。

    返回：
        形状为 [num_streams, n_band, sig_len] 的带限噪声张量。

    算法：
        1. 生成白噪声；
        2. 使用 filter_bank 在频域进行带通滤波；
        3. 根据每条流自己的 T60 只在有效长度内做能量归一化；
        4. 超出 T60 的位置置零。
    '''
    filt = _get_filter_bank(device, dtype)  # 获取目标设备上的滤波器组
    n_band, filter_len = filt.shape  # 读取频带数和滤波器长度
    pad = filter_len // 2  # 与原 NumPy 版本保持一致，前后各留半个滤波器长度的 padding
    white_len = sig_len + 2 * pad  # 白噪声长度 = 目标长度 + 前后 padding
    full_len = white_len + filter_len - 1  # full convolution 的理论长度
    n_fft = 1 << (full_len - 1).bit_length()  # 取不小于 full_len 的 2 次幂 FFT 长度，提高 FFT 效率
    filter_fft = _get_filter_fft(n_fft, device, dtype)  # 获取缓存后的滤波器频域表示
    gen = _get_rng(device)  # 获取固定 seed 的随机数生成器

    chunk_size = int(os.environ.get('SRIR_DEC_NOISE_CHUNK', '32'))  # 分块生成噪声，避免大 batch 显存峰值过高
    noise_out = torch.empty((num_streams, n_band, sig_len), device=device, dtype=dtype)  # 预分配输出噪声张量
    crop_start = filter_len - 1  # 从 full convolution 中裁剪出近似 same 区间的起点

    # --------------------------------------------------------
    # 分块生成白噪声并完成频域卷积
    # --------------------------------------------------------
    for st in range(0, num_streams, chunk_size):  # 按 stream 维度分块，控制显存占用
        ed = min(st + chunk_size, num_streams)  # 当前块的结束位置
        white = torch.randn((ed - st, white_len), device=device, dtype=dtype, generator=gen)  # 生成当前块白噪声
        white_fft = torch.fft.rfft(white, n=n_fft, dim=-1)  # 对白噪声做 rFFT
        conv = torch.fft.irfft(white_fft[:, None, :] * filter_fft[None, :, :], n=n_fft, dim=-1)  # 频域相乘并反变换得到带限噪声
        conv = conv[:, :, crop_start: crop_start + sig_len]  # 裁剪到目标长度 sig_len
        noise_out[st:ed] = conv  # 写入预分配输出张量

    # --------------------------------------------------------
    # 按每条流的有效 T60 做归一化和 mask
    # --------------------------------------------------------
    time_id = torch.arange(sig_len, device=device).view(1, 1, sig_len)  # 构造时间索引 [1, 1, sig_len]
    valid = time_id < t60.view(num_streams, 1, 1)  # 得到每条 stream 的有效时间 mask
    norm = torch.sqrt(torch.sum(noise_out.square() * valid, dim=-1, keepdim=True))  # 只在有效区间内计算每个频带的能量范数
    noise_out = torch.where(valid, noise_out / (norm + _EPS), torch.zeros_like(noise_out))  # 有效区间归一化，padding 区间置零
    return noise_out  # 返回带限且归一化后的噪声


@torch.no_grad()  # 关闭梯度以提高推理效率
def _ev_up_sample_batch(RIR_ev_energy: torch.Tensor, t60: torch.Tensor, max_t60: int) -> torch.Tensor:
    '''
    批量上采样 late energy envelope。

    参数：
        RIR_ev_energy: late 能量包络，形状为 [B, C, n_band, T_ev]。
        t60: 每条样本的真实 T60 采样点数，形状为 [B]。
        max_t60: batch 内最大 T60，用作 padding 后统一输出长度。

    返回：
        形状为 [B, C, n_band, max_t60] 的上采样能量包络。

    与原 ev_up_sample 的对应关系：
        1. 末尾补 20% 的 0；
        2. 对每条样本线性插值到 round(T60 * 1.2)；
        3. 只保留前 T60；
        4. batch 中为了并行，统一 padding 到 max_t60。
    '''
    bsz, ch, n_band, ev_len = RIR_ev_energy.shape  # 读取 batch、通道、频带和原包络长度
    pad_len = round(0.2 * ev_len)  # 计算原逻辑中的 20% 末尾补零长度
    if pad_len > 0:  # 如果补零长度大于 0
        pad = torch.zeros((bsz, ch, n_band, pad_len), device=RIR_ev_energy.device, dtype=RIR_ev_energy.dtype)  # 构造补零张量
        ev_pad = torch.cat((RIR_ev_energy, pad), dim=-1)  # 在时间维末尾拼接零
    else:  # 理论上 ev_len 很小时才可能不补零
        ev_pad = RIR_ev_energy  # 不补零，直接使用原包络

    old_len = ev_pad.shape[-1]  # 补零后的包络长度
    full_len = torch.round(t60.to(RIR_ev_energy.dtype) * 1.2).long().clamp_min(2)  # 每条样本插值目标长度 round(T60*1.2)
    j = torch.arange(max_t60, device=RIR_ev_energy.device, dtype=RIR_ev_energy.dtype).view(1, max_t60)  # 构造统一输出时间索引
    pos = j * ((old_len - 1) / (full_len.to(RIR_ev_energy.dtype).view(bsz, 1) - 1.0))  # 将输出索引映射回旧包络坐标
    pos = pos.clamp(0.0, old_len - 1.0)  # 防止插值坐标越界

    left = torch.floor(pos).long()  # 线性插值左端点索引
    right = torch.clamp(left + 1, max=old_len - 1)  # 线性插值右端点索引
    alpha = (pos - left.to(pos.dtype)).view(bsz, 1, 1, max_t60)  # 线性插值权重，扩展到 [B,1,1,T]

    left = left.view(bsz, 1, 1, max_t60).expand(-1, ch, n_band, -1)  # 将左端点索引扩展到所有通道和频带
    right = right.view(bsz, 1, 1, max_t60).expand(-1, ch, n_band, -1)  # 将右端点索引扩展到所有通道和频带
    y0 = torch.gather(ev_pad, -1, left)  # 从原包络中取左端点值
    y1 = torch.gather(ev_pad, -1, right)  # 从原包络中取右端点值
    out = y0 * (1.0 - alpha) + y1 * alpha  # 按线性插值公式计算输出包络

    valid = torch.arange(max_t60, device=RIR_ev_energy.device).view(1, 1, 1, max_t60) < t60.view(bsz, 1, 1, 1)  # 构造 T60 有效 mask
    return torch.where(valid, out, torch.zeros_like(out))  # 超出各自 T60 的位置置零后返回


@torch.no_grad()  # late 解码只用于推理，不需要梯度
def _late_decoder_batch(reverb_ev: torch.Tensor, t60: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    '''
    批量解码 late reverberation。

    参数：
        reverb_ev: 网络输出的 late energy envelope，形状为 [B, C, n_band, T_ev]。
        t60: 每条样本的 T60 采样点数，形状为 [B]。

    返回：
        rir_late: late reverb 波形，形状为 [B, C, max_t60]。
        valid_mask: 有效时间 mask，形状为 [B, 1, max_t60]。

    处理流程：
        1. 将归一化包络反变换为能量；
        2. 将包络上采样到 T60 对应长度；
        3. 生成每个 batch/channel 的带限噪声；
        4. 用能量包络调制噪声；
        5. 先逐频带归一化，再跨频带求和并整体归一化。
    '''
    bsz, ch, n_band, _ = reverb_ev.shape  # 读取 batch、通道和频带数量
    max_t60 = int(t60.max().item())  # 取 batch 内最大 T60，作为统一 padding 长度
    device, dtype = reverb_ev.device, reverb_ev.dtype  # 记录输入所在设备和精度

    # --------------------------------------------------------
    # 归一化 envelope 反变换到线性能量并上采样
    # --------------------------------------------------------
    ev_energy = torch.pow(torch.tensor(10.0, device=device, dtype=dtype), reverb_ev * 6.0 - 12.0)  # 对应原 10 ** (ev * 6 - 12)
    ev_energy = _ev_up_sample_batch(ev_energy, t60, max_t60)  # 将能量包络上采样到各自 T60，并 padding 到 max_t60
    ev_amp = torch.sqrt(torch.clamp(ev_energy, min=0.0))  # 能量开方得到幅度包络，clamp 防止极小负数

    # --------------------------------------------------------
    # 生成 batch/channel 级别的带限噪声
    # --------------------------------------------------------
    stream_t60 = t60[:, None].expand(bsz, ch).reshape(-1)  # 将每个样本的 T60 扩展到每个通道，得到 [B*C]
    noise = _band_pass_noise_batch(bsz * ch, max_t60, stream_t60, device, dtype)  # 生成 [B*C, n_band, max_t60] 带限噪声
    noise = noise.view(bsz, ch, n_band, max_t60)  # 恢复为 [B, C, n_band, max_t60]

    # --------------------------------------------------------
    # 用 envelope 调制噪声并做能量归一化
    # --------------------------------------------------------
    valid = torch.arange(max_t60, device=device).view(1, 1, 1, max_t60) < t60.view(bsz, 1, 1, 1)  # 构造 [B,1,1,T] 有效 mask
    rir_band = ev_amp * noise  # 将幅度包络乘到带限噪声上，得到每个频带的 late RIR
    band_norm = torch.sqrt(torch.sum(rir_band.square() * valid, dim=-1, keepdim=True))  # 计算每个频带当前波形能量
    band_energy = torch.sqrt(torch.sum(ev_energy * valid, dim=-1, keepdim=True))  # 计算每个频带目标包络能量
    rir_band = torch.where(band_norm > _EPS, rir_band / (band_norm + _EPS) * band_energy, torch.zeros_like(rir_band))  # 将每个频带归一化到目标能量

    # --------------------------------------------------------
    # 跨频带求和并整体匹配 late 总能量
    # --------------------------------------------------------
    rir_late = torch.sum(rir_band, dim=2)  # 沿频带维求和，得到 [B, C, T] late reverb
    valid_ch = valid[:, :, :1, :]  # 将 mask 调整为可用于 [B,C,T] 的形状
    late_norm = torch.sqrt(torch.sum(rir_late.square() * valid_ch.squeeze(2), dim=-1, keepdim=True))  # 计算每个通道 late 波形当前能量
    late_energy = torch.sqrt(torch.sum(ev_energy * valid, dim=(-2, -1), keepdim=True)).squeeze(2)  # 计算每个通道所有频带的目标总能量
    rir_late = torch.where(late_norm > _EPS, rir_late / (late_norm + _EPS) * late_energy, torch.zeros_like(rir_late))  # 将 late 总波形归一化到目标总能量
    return rir_late, valid_ch.squeeze(2)  # 返回 late 波形和有效 mask


@torch.no_grad()  # 拼接过程不需要梯度
def _rir_reverb_connect_batch(RIR_ER: torch.Tensor,
                              reverb: torch.Tensor,
                              t60: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    '''
    批量拼接 ER 和 late reverb。

    参数：
        RIR_ER: early response 波形，形状为 [B, C, er_len]。
        reverb: late reverb 波形，形状为 [B, C, rv_len]。
        t60: 每条样本的 late 有效长度，形状为 [B]。

    返回：
        out: 拼接后的 SRIR，形状为 [B, C, out_len]。
        lengths: 每条样本的真实输出长度，形状为 [B]。

    说明：
        该函数对应原 rir_reverb_contect：
        ER 末尾使用 cos 窗衰减，late 在 ER 尾部使用 sin 窗淡入，从而实现 512 点交叉淡化。
    '''
    bsz, ch, er_len = RIR_ER.shape  # 读取 batch、通道数和 ER 长度
    rv_len = reverb.shape[-1]  # 读取 late reverb 的 padding 长度
    out_len = max(er_len, rv_len)  # 输出长度取 ER 与 late 中较长者
    device, dtype = RIR_ER.device, RIR_ER.dtype  # 记录设备和精度

    # --------------------------------------------------------
    # 构造 ER 衰减窗和 late 淡入窗
    # --------------------------------------------------------
    fade_len = 512  # 与原代码一致，交叉淡化长度固定为 512 点
    fade_start = er_len - fade_len  # 交叉淡化从 ER 末尾前 512 点开始
    theta = torch.linspace(0.0, torch.pi / 2.0, fade_len, device=device, dtype=dtype)  # 构造 0 到 pi/2 的角度序列

    er_win = torch.ones(er_len, device=device, dtype=dtype)  # 初始化 ER 窗函数为全 1
    er_win[fade_start:] = torch.cos(theta)  # ER 末尾使用 cos 从 1 衰减到 0

    rv_win = torch.ones((bsz, out_len), device=device, dtype=dtype)  # 初始化 late 窗函数为全 1
    rv_win[:, :fade_start] = 0.0  # late 在交叉淡化开始之前保持为 0
    rv_win[:, fade_start:er_len] = torch.sin(theta)  # late 在 ER 末尾使用 sin 从 0 淡入到 1
    valid_out = torch.arange(out_len, device=device).view(1, out_len) < torch.maximum(t60, torch.full_like(t60, er_len)).view(bsz, 1)  # 构造每条样本真实输出长度 mask
    rv_win = torch.where(valid_out, rv_win, torch.zeros_like(rv_win))  # 超出真实输出长度的位置不保留 late

    # --------------------------------------------------------
    # 将 late 写入输出，再叠加加窗后的 ER
    # --------------------------------------------------------
    out = torch.zeros((bsz, ch, out_len), device=device, dtype=dtype)  # 初始化输出 SRIR 张量
    out[:, :, :rv_len] = reverb  # 先写入 late reverb
    out = out * rv_win[:, None, :]  # 对 late 部分施加淡入窗和有效长度 mask
    out[:, :, :er_len] = out[:, :, :er_len] + RIR_ER * er_win.view(1, 1, er_len)  # 将加窗 ER 叠加到输出前 er_len 区间
    out = torch.where(valid_out[:, None, :], out, torch.zeros_like(out))  # 输出真实长度之外统一置零
    return out, torch.maximum(t60, torch.full_like(t60, er_len))  # 返回输出和每条样本真实长度


@torch.no_grad()  # 整体 SRIR 解码只用于推理，不需要梯度
def SRIR_decode(SRIR_ER,
                para,
                reverb_ev,
                LoR,
                return_lengths: bool = False,
                dtype: torch.dtype = torch.float32,
                with_lor: Optional[bool] = None,
                io_rir_er_type: Optional[str] = None,
                device: Optional[Union[str, torch.device]] = None):
    '''
    GPU/CPU 并行版 SRIR 解码器。

    输入：
        SRIR_ER:
            单样本为 [C, 4096]，batch 为 [B, C, 4096]。
        para:
            单样本为 [3]，batch 为 [B, 3]。
            para[:, 0] 为 T60 秒数，para[:, 1] 为 ER 能量，para[:, 2] 为 late 能量。
        reverb_ev:
            单样本为 [C, n_band, T_ev]，batch 为 [B, C, n_band, T_ev]。
        LoR:
            单样本为 [C, 4096]，batch 为 [B, C, 4096]。
        return_lengths:
            是否同时返回每条样本的真实输出长度。
        dtype:
            解码使用的数据类型，默认 torch.float32。

    返回：
        若 return_lengths=False：
            返回 padding 后的 SRIR。
        若 return_lengths=True：
            返回 (SRIR, lengths)，其中 lengths 为每条样本真实输出长度。

    兼容性：
        1. 输入是 torch.Tensor 时返回 torch.Tensor；
        2. 输入全是 NumPy/列表时返回 NumPy 数组；
        3. 单样本输入会恢复为单样本输出，不额外保留 batch 维。
    '''
    input_is_torch = any(isinstance(x, torch.Tensor) for x in (SRIR_ER, para, reverb_ev, LoR))  # 判断是否至少有一个输入为 torch.Tensor
    device = torch.device(device) if device is not None else None  # 若调用方指定设备，则转换为 torch.device
    SRIR_ER, para, reverb_ev, LoR, single_input = _to_batch_inputs(SRIR_ER, para, reverb_ev, LoR, dtype, device)  # 统一整理为 batch 张量
    device = SRIR_ER.device  # 记录当前解码设备

    # --------------------------------------------------------
    # 从 para 中读取 T60、ER 能量和 late 能量
    # --------------------------------------------------------
    t60 = torch.round(para[:, 0] * 48000.0).long().clamp_min(10)  # 将 T60 秒数转换为采样点，并设置最小长度 10
    RIR_ER_en = para[:, 1].view(-1, 1, 1)  # 取 ER 目标能量，并扩展为 [B,1,1]
    reverb_en = para[:, 2].view(-1, 1, 1)  # 取 late 目标能量，并扩展为 [B,1,1]

    # --------------------------------------------------------
    # 根据配置决定是否使用 LoR
    # --------------------------------------------------------
    io_type, use_lor = _runtime_flags(io_rir_er_type=io_rir_er_type, with_lor=with_lor)  # 读取 IO_RIR_ER_type 和 with_LoR 开关
    if (io_type == 'RIR') or (not use_lor):  # 若输入 ER 已包含完整 RIR 或测试不启用 LoR
        LoR = LoR * 0.0  # 将 LoR 置零，保持原 SRERS_decoder 行为

    # --------------------------------------------------------
    # 恢复 ER 能量并叠加 LoR
    # --------------------------------------------------------
    er_norm = torch.sqrt(torch.sum(SRIR_ER.square(), dim=(1, 2), keepdim=True))  # 计算每条样本 ER 的全通道能量范数
    RIR_ER = torch.where(er_norm > _EPS, SRIR_ER / (er_norm + _EPS) * RIR_ER_en, torch.zeros_like(SRIR_ER))  # 将归一化 ER 恢复到目标能量
    RIR_ER = RIR_ER + 0.95 * LoR  # 按原逻辑叠加 0.95 倍 LoR

    # --------------------------------------------------------
    # 解码 late reverb 并恢复 late 总能量
    # --------------------------------------------------------
    reverb, _ = _late_decoder_batch(reverb_ev, t60)  # 根据 envelope 和随机带限噪声生成 late reverb
    rv_len = reverb.shape[-1]  # 读取 late reverb 的 padding 长度
    rv_valid = torch.arange(rv_len, device=device).view(1, 1, rv_len) < t60.view(-1, 1, 1)  # 构造 late 有效时间 mask
    rv_norm = torch.sqrt(torch.sum(reverb.square() * rv_valid, dim=(1, 2), keepdim=True))  # 计算每条样本 late 当前能量
    reverb = torch.where(rv_norm > _EPS, reverb / (rv_norm + _EPS) * reverb_en, torch.zeros_like(reverb))  # 将 late 恢复到 para 给定目标能量

    # --------------------------------------------------------
    # 拼接 ER 和 late，得到最终 SRIR
    # --------------------------------------------------------
    RIR, lengths = _rir_reverb_connect_batch(RIR_ER, reverb, t60)  # 使用 512 点交叉淡化拼接 ER 和 late

    # --------------------------------------------------------
    # 单样本输入恢复为单样本输出
    # --------------------------------------------------------
    if single_input:  # 如果输入原本是 [C,T] 形式的单样本
        RIR = RIR[0, :, :int(lengths[0].item())]  # 去掉 batch 维，并裁剪到真实输出长度
        lengths = lengths[:1]  # 保留长度张量，便于 return_lengths=True 时返回

    # --------------------------------------------------------
    # 根据输入类型决定返回 torch.Tensor 还是 NumPy
    # --------------------------------------------------------
    if input_is_torch:  # 如果输入中存在 torch.Tensor
        return (RIR, lengths) if return_lengths else RIR  # 保持 torch 输出，避免推理阶段提前转 CPU

    RIR_np = RIR.detach().cpu().numpy()  # 如果输入是 NumPy/列表，则转回 NumPy 输出
    lengths_np = lengths.detach().cpu().numpy()  # 长度也转回 NumPy
    return (RIR_np, lengths_np) if return_lengths else RIR_np  # 按 return_lengths 选择返回形式


def decode_predicted_para(fake_para: torch.Tensor) -> torch.Tensor:
    '''
    将网络 PA 输出还原为 RIR 重构所需的物理能量尺度。

    参数：
        fake_para: 网络输出的参数张量，形状为 [B, 3]。

    返回：
        para: 与 fake_para 同形状的张量，其中第 1、2 维能量参数执行 10 ** (x - 3) 反变换。
    '''
    para = fake_para.clone()  # 避免原地修改网络输出
    scale_base = torch.tensor(10.0, device=para.device, dtype=para.dtype)  # 构造与输入同设备同精度的底数
    para[:, 1:] = torch.pow(scale_base, para[:, 1:] - 3.0)  # 将 ER / late 能量参数还原到物理尺度
    return para  # 返回可直接送入 SRIR_decode 的参数


# --------------------------------------------------------
# 兼容原文件的单样本函数名
# --------------------------------------------------------
def SRERS_decoder(SRIR_ER, para, reverb_ev, LoR):
    '''
    兼容原工程调用的单样本 SRERS_decoder 接口。

    参数：
        SRIR_ER: early response 参数化波形。
        para: [T60, ER_energy, late_energy]。
        reverb_ev: late energy envelope。
        LoR: low-order reflection 波形。

    返回：
        解码后的 SRIR。

    说明：
        内部直接调用新的 SRIR_decode，因此既兼容 NumPy 单样本，也兼容 torch 张量输入。
    '''
    return SRIR_decode(SRIR_ER, para, reverb_ev, LoR, return_lengths=False)  # 调用统一解码入口并保持原函数返回形式


def ev_up_sample(RIR_ev_array, T60):
    '''
    兼容原工程调用的单样本 envelope 上采样接口。

    参数：
        RIR_ev_array: 单条 late energy envelope，形状通常为 [n_band, T_ev]。
        T60: 目标 T60 采样点数。

    返回：
        上采样后的 NumPy 数组。

    说明：
        该函数主要用于兼容旧代码；推理批处理时应优先使用 _ev_up_sample_batch。
    '''
    RIR_ev = _as_tensor(RIR_ev_array, dtype=torch.float32)  # 将输入 envelope 转为 float32 张量
    RIR_ev = RIR_ev.unsqueeze(0).unsqueeze(0)  # 添加 batch 和 channel 维，变为 [1,1,n_band,T_ev]
    t60 = torch.as_tensor([max(int(T60), 10)], device=RIR_ev.device, dtype=torch.long)  # 构造长度至少为 10 的 T60 张量
    out = _ev_up_sample_batch(RIR_ev, t60, int(t60.max().item()))[0, 0]  # 调用 batch 上采样后取回单样本单通道结果
    return out.detach().cpu().numpy()  # 转回 NumPy，保持旧接口输出类型


def late_decoder(T60, RIR_ev_array):
    '''
    兼容原工程调用的单样本 late_decoder 接口。

    参数：
        T60: late reverb 的目标长度，单位为采样点。
        RIR_ev_array: 单通道 late energy envelope，形状通常为 [n_band, T_ev]。

    返回：
        单通道 late reverb 的 NumPy 波形。

    说明：
        该函数内部调用 GPU/CPU batch 版本，但输出保持旧版 NumPy 形式。
    '''
    RIR_ev = _as_tensor(RIR_ev_array, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # 转张量并添加 batch/channel 维
    t60 = torch.as_tensor([max(int(T60), 10)], device=RIR_ev.device, dtype=torch.long)  # 构造 T60 张量并限制最小长度
    out, _ = _late_decoder_batch(RIR_ev, t60)  # 调用 batch late 解码器
    return out[0, 0].detach().cpu().numpy()  # 取回单样本单通道 late 波形并转 NumPy


def rir_reverb_contect(RIR_ER, reverb):
    '''
    兼容原工程拼写的 rir_reverb_contect 接口。

    参数：
        RIR_ER: early response 波形，形状为 [C, T_er]。
        reverb: late reverb 波形，形状为 [C, T_late]。

    返回：
        拼接后的 SRIR NumPy 数组。

    说明：
        函数名保留原拼写 contect，不改为 connect，以避免破坏项目中已有调用。
    '''
    RIR_ER_t = _as_tensor(RIR_ER, dtype=torch.float32).unsqueeze(0)  # 将 ER 转为 [1,C,T_er] 张量
    reverb_t = _as_tensor(reverb, RIR_ER_t.device, torch.float32).unsqueeze(0)  # 将 late 转到同一设备并添加 batch 维
    t60 = torch.as_tensor([reverb_t.shape[-1]], device=RIR_ER_t.device, dtype=torch.long)  # 使用 late 长度作为 T60
    out, _ = _rir_reverb_connect_batch(RIR_ER_t, reverb_t, t60)  # 调用 batch 拼接函数
    return out[0].detach().cpu().numpy()  # 去掉 batch 维并转回 NumPy


def band_pass_noise(sig_len):
    '''
    兼容原工程调用的单样本 band_pass_noise 接口。

    参数：
        sig_len: 输出噪声长度，单位为采样点。

    返回：
        形状为 [n_band, sig_len] 的 NumPy 带限噪声。

    说明：
        该函数主要用于旧代码或调试；推理批处理时应优先调用 _band_pass_noise_batch。
    '''
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 优先在 GPU 上生成噪声
    t60 = torch.as_tensor([int(sig_len)], device=device, dtype=torch.long)  # 构造单条噪声流的有效长度
    out = _band_pass_noise_batch(1, int(sig_len), t60, device, torch.float32)[0]  # 批量生成后取第 0 条 stream
    return out.detach().cpu().numpy()  # 转回 NumPy，保持旧接口输出类型
