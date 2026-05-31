"""Spatial GSLR metrics without dataset scanning or batch file loading."""

import math
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F

@dataclass
class SpatialMetricConfig:
    """空间事件指标计算配置。"""

    sample_rate: int = 48000  # 采样率，默认与项目一致为 48 kHz
    early_len: int = 4096  # 早期段长度，默认仅对前 4096 点做 impulse / SDM / DPOT 分解
    sound_speed: float = 343.0  # 声速，单位 m/s
    input_format: str = 'A'  # 输入格式，默认 A 表示 FOA-A-format 四胶囊原始通道
    metric_weight_mode: str = 'log'  # 指标权重模式，可选 'none'、'linear'、'log'
    log_weight_reference_energy: float = 30.0  # 对数能量权重的偏置，避免能量很小时权重为负
    eps: float = 1e-10  # 数值稳定项，避免除零和 log(0)

    # --------------------------------------------------------
    # impulse / onset / SDM 参数
    # --------------------------------------------------------
    max_events: int = 256  # 每条 SRIR 最多保留的 impulse 事件数量
    impulse_env_win: int = 5  # 四通道联合能量包络的平滑窗口长度，必须为奇数
    impulse_min_sep: int = 9  # impulse 局部极大值的最小间隔，必须为奇数
    impulse_db_threshold: float = -40.0  # impulse 检测相对阈值，单位 dB，参考当前 SRIR 的最大联合包络
    onset_search_pre: int = 24  # 每个 impulse 峰值前用于 onset 搜索的采样点数
    onset_search_post: int = 24  # 每个 impulse 峰值后用于 onset 搜索的采样点数
    onset_env_win: int = 3  # 单通道 onset 局部能量包络平滑窗口，必须为奇数
    onset_rel_threshold: float = 0.05  # onset 阈值，取当前事件当前通道局部峰值的相对比例
    doa_max_residual_samples: float = 2.5  # TDOA 反演 DOA 后允许的最大平面波拟合残差，单位 sample

    # --------------------------------------------------------
    # DPOT-SDM 参数
    # --------------------------------------------------------
    dpot_sinkhorn_iters: int = 60  # Sinkhorn 迭代次数，越大越接近最优传输解但越慢
    dpot_sinkhorn_reg_m: float = 0.25  # 熵正则系数，单位 m，用于稳定 batched Sinkhorn
    dpot_unmatched_cost_m: float = 8.0  # 事件未匹配时进入 dustbin 的惩罚，单位 m
    dpot_dummy_mass: float = 1.0  # dustbin 额外质量，用于允许反射缺失、伪反射或数量不一致
    matched_mass_min: float = 1e-4  # 判断 onset-TDOA / onset-DOA 误差是否有效的最小匹配质量

    # --------------------------------------------------------
    # IACC 参数
    # --------------------------------------------------------
    iacc_lag_ms: float = 1.0  # IACC 最大互相关搜索的单侧滞后范围，默认采用 ±1 ms

    # --------------------------------------------------------
    # 指标计算前的 FIR 带通滤波配置
    # --------------------------------------------------------
    apply_metric_bandpass: bool = True  # 是否在计算空间指标前执行带通滤波
    metric_bandpass_low_hz: float = 200.0  # 带通下截止频率，单位 Hz
    metric_bandpass_high_hz: float = 12000.0  # 带通上截止频率，单位 Hz
    metric_fir_num_taps: int = 257  # FIR 滤波器长度，必须为奇数
    metric_fir_normalize_hz: float = 1000.0  # FIR 通带幅值归一化参考频率，单位 Hz



class SpatialMetricComputer(torch.nn.Module):
    """基于 Torch 的 FOA-A-format SRIR onset-TDOA / SDM / DPOT / IACC 指标计算模块。"""

    def __init__(self, metric_cfg: SpatialMetricConfig):
        super().__init__()  # 初始化父类 nn.Module
        self.metric_cfg = metric_cfg  # 保存空间指标计算配置

        if metric_cfg.impulse_env_win % 2 != 1:  # 检查 impulse 包络窗口是否为奇数
            raise ValueError('impulse_env_win must be odd.')  # 抛出配置错误
        if metric_cfg.impulse_min_sep % 2 != 1:  # 检查 impulse 局部极大值窗口是否为奇数
            raise ValueError('impulse_min_sep must be odd.')  # 抛出配置错误
        if metric_cfg.onset_env_win % 2 != 1:  # 检查 onset 包络窗口是否为奇数
            raise ValueError('onset_env_win must be odd.')  # 抛出配置错误
        if metric_cfg.metric_weight_mode.lower() not in ('none', 'linear', 'log'):  # 检查权重模式是否合法
            raise ValueError("metric_weight_mode must be 'none', 'linear', or 'log'.")  # 抛出配置错误

        a_to_foa = 0.5 * torch.tensor([  # 定义常见四面体 A-format 到 FOA WXYZ 的线性变换
            [1.0, 1.0, 1.0, 1.0],  # W 通道变换权重
            [1.0, 1.0, -1.0, -1.0],  # X 通道变换权重
            [1.0, -1.0, -1.0, 1.0],  # Y 通道变换权重
            [1.0, -1.0, 1.0, -1.0],  # Z 通道变换权重
        ], dtype=torch.float32)  # 设置矩阵数据类型为 float32
        self.register_buffer('a_to_foa', a_to_foa, persistent=False)  # 注册 A-format 到 FOA 的非持久化 buffer

        microphone_positions_np = np.array([  # 定义四面体 A-format 胶囊的几何坐标
            [1.0, 1.0, 1.0],  # 第 0 个胶囊方向坐标
            [1.0, -1.0, -1.0],  # 第 1 个胶囊方向坐标
            [-1.0, -1.0, 1.0],  # 第 2 个胶囊方向坐标
            [-1.0, 1.0, -1.0],  # 第 3 个胶囊方向坐标
        ], dtype=np.float32) * 0.06  # 将无量纲四面体坐标缩放为项目中原脚本使用的阵列半径尺度
        microphone_positions = torch.from_numpy(microphone_positions_np)  # 将 NumPy 坐标转为 Torch 张量
        self.register_buffer('microphone_positions', microphone_positions, persistent=False)  # 注册麦克风位置 buffer

        pair_indices = torch.tensor([  # 定义 4 通道之间全部 6 组 pair
            [0, 1],  # 第 0 组通道对
            [0, 2],  # 第 1 组通道对
            [0, 3],  # 第 2 组通道对
            [1, 2],  # 第 3 组通道对
            [1, 3],  # 第 4 组通道对
            [2, 3],  # 第 5 组通道对
        ], dtype=torch.long)  # 设置通道索引为 long 类型
        self.register_buffer('pair_indices', pair_indices, persistent=False)  # 注册通道对索引 buffer

        pair_position_diff = microphone_positions[pair_indices[:, 0]] - microphone_positions[pair_indices[:, 1]]  # 计算每组通道对的位置差 r_i-r_j
        self.register_buffer('pair_position_diff', pair_position_diff, persistent=False)  # 注册位置差矩阵 [6, 3]

        pair_position_pinv = torch.linalg.pinv(pair_position_diff)  # 计算位置差矩阵的伪逆 [3, 6]
        self.register_buffer('pair_position_pinv', pair_position_pinv, persistent=False)  # 注册伪逆矩阵，用于由 TDOA 反演方向

        self.iacc_lag_samples = max(1, int(round(metric_cfg.iacc_lag_ms * metric_cfg.sample_rate / 1000.0)))  # 将 IACC 滞后范围转换为采样点

        metric_bandpass_fir_kernel = self._build_metric_bandpass_fir_kernel()  # 构造指标计算前使用的 FIR 带通滤波器
        self.register_buffer('metric_bandpass_fir_kernel', metric_bandpass_fir_kernel, persistent=False)  # 注册 FIR 核 buffer
        self._metric_fir_fft_response_cache = {}  # 初始化 FIR 频响缓存，避免重复 FFT

    # --------------------------------------------------------
    # FIR 带通滤波
    # --------------------------------------------------------
    def _build_metric_bandpass_fir_kernel(self) -> torch.Tensor:
        """
        构造 200–8000 Hz 线性相位 FIR 带通滤波器。\n
        设计方式：\n
        1. 使用两个理想低通滤波器之差构造理想带通；\n
        2. 使用 Hann 窗截断以降低旁瓣；\n
        3. 在指定通带参考频率处归一化增益；\n
        4. 保持奇数长度以获得整数采样中心延迟。\n
        """
        num_taps = int(self.metric_cfg.metric_fir_num_taps)  # 读取 FIR 核长度
        if num_taps % 2 == 0:  # 若核长为偶数
            raise ValueError('metric_fir_num_taps must be odd for a symmetric linear-phase FIR filter.')  # 抛出配置错误

        sample_rate = float(self.metric_cfg.sample_rate)  # 读取采样率
        low_hz = float(self.metric_cfg.metric_bandpass_low_hz)  # 读取低截止频率
        high_hz = float(self.metric_cfg.metric_bandpass_high_hz)  # 读取高截止频率

        if not (0.0 < low_hz < high_hz < sample_rate / 2.0):  # 检查截止频率是否合法
            raise ValueError('Bandpass cutoff frequencies must satisfy 0 < low < high < Nyquist.')  # 抛出配置错误

        center = (num_taps - 1) / 2.0  # 计算 FIR 对称中心位置
        n = torch.arange(num_taps, dtype=torch.float32)  # 构造离散采样索引
        m = n - center  # 将索引平移到以中心为零的位置

        high_norm = high_hz / sample_rate  # 将高截止频率归一化到采样率
        low_norm = low_hz / sample_rate  # 将低截止频率归一化到采样率
        lowpass_high = 2.0 * high_norm * torch.sinc(2.0 * high_norm * m)  # 构造 high_hz 理想低通脉冲响应
        lowpass_low = 2.0 * low_norm * torch.sinc(2.0 * low_norm * m)  # 构造 low_hz 理想低通脉冲响应
        fir_kernel = lowpass_high - lowpass_low  # 两个低通响应相减得到带通响应

        window = torch.hann_window(num_taps, periodic=False, dtype=torch.float32)  # 构造非周期 Hann 窗
        fir_kernel = fir_kernel * window  # 对理想带通核加窗截断

        normalize_hz = float(self.metric_cfg.metric_fir_normalize_hz)  # 读取通带归一化参考频率
        if not (low_hz < normalize_hz < high_hz):  # 检查参考频率是否在通带内
            raise ValueError('metric_fir_normalize_hz must lie inside the bandpass range.')  # 抛出配置错误

        omega = 2.0 * math.pi * normalize_hz / sample_rate  # 计算参考频率的离散角频率
        response_real = torch.sum(fir_kernel * torch.cos(omega * m))  # 计算对称 FIR 在参考频率处的实数频响
        gain = torch.abs(response_real).clamp_min(self.metric_cfg.eps)  # 取参考频率增益并避免除零
        fir_kernel = fir_kernel / gain  # 将通带参考频率处的增益归一化到约 1

        return fir_kernel  # 返回形状为 [num_taps] 的 FIR 核

    @staticmethod
    def _next_power_of_two(value: int) -> int:
        """返回不小于 value 的最小 2 的整数次幂。"""
        if value <= 1:  # 若输入小于等于 1
            return 1  # 返回 1
        return 1 << (value - 1).bit_length()  # 使用 bit_length 快速计算 2 次幂

    def _get_metric_fir_fft_response(self, n_fft: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """获取指定 FFT 长度、设备和 dtype 下的 FIR 频率响应。"""
        cache_key = (int(n_fft), str(device), str(dtype))  # 以 FFT 长度、设备和 dtype 作为缓存键
        if cache_key not in self._metric_fir_fft_response_cache:  # 若当前频响尚未缓存
            fir_kernel = self.metric_bandpass_fir_kernel.to(device=device, dtype=dtype)  # 将 FIR 核迁移到目标设备并匹配 dtype
            fir_fft_response = torch.fft.rfft(fir_kernel, n=n_fft, dim=-1)  # 计算 FIR 核的单边频响
            self._metric_fir_fft_response_cache[cache_key] = fir_fft_response  # 写入缓存字典
        return self._metric_fir_fft_response_cache[cache_key]  # 返回缓存的 FIR 频响

    def metric_bandpass_filter(self, x: torch.Tensor) -> torch.Tensor:
        """
        使用 FFT 线性卷积执行指标前带通滤波。\n
        输入为 [B, 4, T]，输出仍为 [B, 4, T]。\n
        采用中心裁剪抵消线性相位 FIR 的群延迟，避免改变 SRIR 的整体时间对齐。\n
        """
        if not self.metric_cfg.apply_metric_bandpass:  # 若关闭带通滤波
            return x.float()  # 直接返回 float32 输入

        x = x.float()  # 将输入统一为 float32
        signal_len = x.shape[-1]  # 读取原始 SRIR 长度
        kernel_len = int(self.metric_bandpass_fir_kernel.numel())  # 读取 FIR 核长度
        full_conv_len = signal_len + kernel_len - 1  # 计算完整线性卷积长度
        n_fft = self._next_power_of_two(full_conv_len)  # 选择不小于完整卷积长度的 2 次幂 FFT 长度

        x_fft = torch.fft.rfft(x, n=n_fft, dim=-1)  # 对输入做实数 FFT，得到 [B, 4, F]
        fir_fft_response = self._get_metric_fir_fft_response(n_fft=n_fft, device=x.device, dtype=x.dtype)  # 获取 FIR 频响
        y_fft = x_fft * fir_fft_response.view(1, 1, -1)  # 频域相乘，相当于时域线性卷积
        y_full = torch.fft.irfft(y_fft, n=n_fft, dim=-1)[..., :full_conv_len]  # 逆 FFT 得到完整线性卷积结果

        center_delay = (kernel_len - 1) // 2  # 计算对称 FIR 的中心群延迟
        y_same = y_full[..., center_delay:center_delay + signal_len]  # 中心裁剪，使输出长度与输入一致
        return y_same  # 返回时间对齐后的带通 SRIR

    # --------------------------------------------------------
    # 基础工具函数
    # --------------------------------------------------------
    @staticmethod
    def _same_avg_pool1d(x: torch.Tensor, kernel_size: int) -> torch.Tensor:
        """执行 same 长度的一维平均池化。"""
        return F.avg_pool1d(x, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)  # 返回同长度平均池化结果

    @staticmethod
    def _same_max_pool1d(x: torch.Tensor, kernel_size: int) -> torch.Tensor:
        """执行 same 长度的一维最大池化。"""
        return F.max_pool1d(x, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)  # 返回同长度最大池化结果

    @staticmethod
    def _gather_1d(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        """从 [B, T] 张量中按 [B, K] 索引收集采样值。"""
        return torch.gather(x, dim=1, index=idx)  # 返回 [B, K] 的采样值

    def _gather_windows(self, x: torch.Tensor, centers: torch.Tensor, pre: int, post: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """围绕 [B, K] 中心索引从 [B, C, T] 中收集局部窗口。"""
        B, C, T = x.shape  # 读取 batch、通道数和时间长度
        K = centers.shape[1]  # 读取每条 SRIR 的事件数量上限
        offsets = torch.arange(-pre, post + 1, device=x.device, dtype=torch.long)  # 构造窗口内相对偏移
        idx = centers[..., None] + offsets[None, None, :]  # 生成 [B, K, W] 的绝对索引
        valid_mask = (idx >= 0) & (idx < T)  # 记录窗口索引是否处于有效时间范围内
        idx = idx.clamp(0, T - 1)  # 将越界索引裁剪到合法范围，避免 gather 报错
        idx_expanded = idx[:, :, None, :].expand(B, K, C, idx.shape[-1])  # 扩展索引到 [B, K, C, W]
        x_expanded = x[:, None, :, :].expand(B, K, C, T)  # 扩展输入到 [B, K, C, T]，expand 不实际复制数据
        windows = torch.gather(x_expanded, dim=-1, index=idx_expanded)  # 收集局部窗口 [B, K, C, W]
        windows = windows * valid_mask[:, :, None, :].to(dtype=windows.dtype)  # 将越界位置置零
        return windows, valid_mask  # 返回局部窗口和有效掩码

    def raw_rir_energy(self, x: torch.Tensor) -> torch.Tensor:
        """计算每条四通道 SRIR 的原始线性能量，返回 [B]。"""
        return torch.sum(x.float() ** 2, dim=(1, 2))  # 对通道维和时间维求平方和

    def aggregation_weight(self, x: torch.Tensor) -> torch.Tensor:
        """根据 metric_weight_mode 返回数据集级指标汇总权重。"""
        mode = self.metric_cfg.metric_weight_mode.lower()  # 读取统一的权重模式
        if mode == 'none':  # 若选择不加权
            return torch.ones(x.shape[0], device=x.device, dtype=torch.float32)  # 返回逐样本等权权重
        energy = torch.sum(x.float() ** 2, dim=(1, 2)).clamp_min(self.metric_cfg.eps)  # 计算每条 SRIR 的线性能量
        if mode == 'linear':  # 若选择线性能量加权
            return energy  # 直接返回线性能量权重
        log_weight = 10.0 * torch.log10(energy) + float(self.metric_cfg.log_weight_reference_energy)  # 计算对数能量权重
        return torch.clamp(log_weight, min=0.0)  # 将负权重截断为 0

    def event_mass_from_energy(self, event_energy: torch.Tensor, event_mask: torch.Tensor) -> torch.Tensor:
        """根据 metric_weight_mode 返回事件级 DPOT 质量。"""
        mode = self.metric_cfg.metric_weight_mode.lower()  # 读取统一的权重模式
        if mode == 'none':  # 若选择不加权
            event_mass = event_mask.to(dtype=event_energy.dtype)  # 每个有效事件质量相同
        elif mode == 'linear':  # 若选择线性能量加权
            event_mass = event_energy.clamp_min(self.metric_cfg.eps) * event_mask.to(dtype=event_energy.dtype)  # 使用线性能量作为事件质量
        else:  # 若选择对数能量加权
            event_mass = 10.0 * torch.log10(event_energy.clamp_min(self.metric_cfg.eps)) + float(self.metric_cfg.log_weight_reference_energy)  # 计算事件对数能量权重
            event_mass = torch.clamp(event_mass, min=0.0) * event_mask.to(dtype=event_energy.dtype)  # 截断负权重并屏蔽无效事件
        return event_mass  # 返回 [B, K] 事件质量

    # --------------------------------------------------------
    # impulse-wise onset-TDOA / SDM 分解
    # --------------------------------------------------------
    def decompose_impulse_sdm(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        对 FOA-A-format 四通道 SRIR 的前 4096 点进行 impulse-wise onset-TDOA / SDM 分解。\n
        输出包括：\n
        1. impulse 峰值位置；\n
        2. 每个 impulse 在四个 A-format capsule 上的 onset 时间；\n
        3. 每个 impulse 的 6 组 onset-TDOA；\n
        4. 每个 impulse 的 DOA 单位向量；\n
        5. 每个 impulse 的局部能量；\n
        6. 每个 impulse 的虚拟声源坐标，用于 DPOT-SDM。\n
        """
        if x.ndim != 3 or x.shape[1] != 4:  # 检查输入是否为 [B, 4, T]
            raise ValueError('A-format SRIR metrics require input shape [B, 4, T].')  # 抛出维度错误

        x = x.float()  # 将输入统一为 float32
        B, C, T_all = x.shape  # 读取 batch、通道数和原始长度
        early_len = min(self.metric_cfg.early_len, T_all)  # 计算实际早期段长度
        x_early = x[:, :, :early_len]  # 截取前 4096 点或更短早期段
        T = x_early.shape[-1]  # 读取实际早期段长度
        dtype = x_early.dtype  # 读取输入 dtype

        capsule_rms = torch.sqrt(torch.mean(x_early.pow(2), dim=1) + self.metric_cfg.eps)  # 计算四通道联合 RMS 包络 [B, T]
        event_env = torch.sqrt(self._same_avg_pool1d(capsule_rms[:, None, :].pow(2), self.metric_cfg.impulse_env_win).squeeze(1) + self.metric_cfg.eps)  # 平滑联合包络 [B, T]

        time_axis = torch.arange(T, device=x.device, dtype=dtype)  # 构造时间索引用于打破平台峰并偏向更早事件
        time_bias = 1.0 - time_axis / max(T - 1, 1)  # 构造随时间递减的极小偏置
        global_peak = event_env.amax(dim=-1, keepdim=True).clamp_min(self.metric_cfg.eps)  # 计算每条 SRIR 的最大联合包络
        nms_score = event_env + global_peak * 1e-6 * time_bias[None, :]  # 添加极小时间偏置以稳定局部峰排序
        local_max = self._same_max_pool1d(nms_score[:, None, :], self.metric_cfg.impulse_min_sep).squeeze(1)  # 计算局部最大值
        is_local_peak = nms_score == local_max  # 标记局部极大值位置
        rel_thr = 10.0 ** (float(self.metric_cfg.impulse_db_threshold) / 20.0)  # 将 dB 阈值转换为幅度比例
        keep = is_local_peak & (event_env >= global_peak * rel_thr) & (global_peak > math.sqrt(self.metric_cfg.eps))  # 得到有效 impulse 候选掩码

        scores = torch.where(keep, event_env, torch.full_like(event_env, float('-inf')))  # 将无效位置分数置为 -inf
        K = min(int(self.metric_cfg.max_events), T)  # 计算实际保留的事件数量上限
        event_strength_unsorted, event_peak_idx_unsorted = torch.topk(scores, k=K, dim=-1, largest=True, sorted=False)  # 取能量最高的 K 个事件候选
        event_mask_unsorted = torch.isfinite(event_strength_unsorted)  # 根据 topk 分数是否有限判断事件是否有效
        sort_key = torch.where(event_mask_unsorted, event_peak_idx_unsorted, torch.full_like(event_peak_idx_unsorted, T))  # 将无效事件排序到末尾
        time_order = torch.argsort(sort_key, dim=-1)  # 按事件时间从早到晚排序
        event_peak_idx = torch.gather(event_peak_idx_unsorted, dim=1, index=time_order)  # 得到按时间排序的事件峰值索引 [B, K]
        event_strength = torch.gather(event_strength_unsorted, dim=1, index=time_order)  # 得到按时间排序的事件强度 [B, K]
        event_mask = torch.gather(event_mask_unsorted, dim=1, index=time_order)  # 得到按时间排序的事件有效掩码 [B, K]
        event_peak_idx = torch.where(event_mask, event_peak_idx, torch.zeros_like(event_peak_idx))  # 将无效事件索引置零
        event_strength = torch.where(event_mask, event_strength, torch.zeros_like(event_strength))  # 将无效事件强度置零

        local_windows, local_valid = self._gather_windows(x_early, event_peak_idx, self.metric_cfg.onset_search_pre, self.metric_cfg.onset_search_post)  # 收集每个事件的局部搜索窗口
        B, K, C, W = local_windows.shape  # 读取局部窗口张量维度
        local_energy = local_windows.abs().reshape(B * K * C, 1, W).pow(2)  # 将局部窗口整理为池化输入并取能量
        local_env = torch.sqrt(self._same_avg_pool1d(local_energy, self.metric_cfg.onset_env_win).reshape(B, K, C, W) + self.metric_cfg.eps)  # 计算单通道局部 onset 包络
        ch_peak, ch_peak_idx = torch.max(local_env, dim=-1)  # 找到每个事件每个通道的局部峰值与位置
        ch_thr = ch_peak * float(self.metric_cfg.onset_rel_threshold)  # 计算 onset 相对阈值
        pos = torch.arange(W, device=x.device, dtype=torch.long).view(1, 1, 1, W)  # 构造局部窗口内位置索引
        before_peak = pos <= ch_peak_idx[..., None]  # 限制 onset 搜索只发生在通道峰值之前或峰值处
        above = (local_env >= ch_thr[..., None]) & before_peak & local_valid[:, :, None, :]  # 标记超过阈值且位于峰值前的候选位置
        prev_above = F.pad(above[..., :-1], pad=(1, 0), value=False)  # 构造前一采样点是否已经超过阈值的掩码
        rising = above & (~prev_above)  # 标记首次越过阈值的上升沿位置
        has_above = above.any(dim=-1)  # 判断每个事件每个通道是否存在可用 onset
        has_rising = rising.any(dim=-1)  # 判断每个事件每个通道是否存在清晰上升沿
        idx_rising = rising.to(torch.float32).argmax(dim=-1)  # 获取上升沿位置索引
        idx_above = above.to(torch.float32).argmax(dim=-1)  # 获取首次超过阈值的位置索引
        onset_local_idx = torch.where(has_rising, idx_rising, idx_above).to(torch.long)  # 优先使用上升沿，否则退化为首次过阈位置

        idx_prev = torch.clamp(onset_local_idx - 1, min=0)  # 取 onset 前一个采样点索引
        e_cur = torch.gather(local_env, dim=-1, index=onset_local_idx[..., None]).squeeze(-1)  # 获取 onset 当前位置包络值
        e_prev = torch.gather(local_env, dim=-1, index=idx_prev[..., None]).squeeze(-1)  # 获取 onset 前一位置包络值
        frac = (ch_thr - e_prev) / (e_cur - e_prev + self.metric_cfg.eps)  # 线性插值估计阈值 crossing 的分数采样位置
        frac = torch.clamp(frac, min=0.0, max=1.0)  # 将分数偏移限制到 [0, 1]
        onset_local_float = idx_prev.to(dtype) + frac.to(dtype)  # 得到局部窗口内的分数 onset 位置
        onset_local_float = torch.where(onset_local_idx > 0, onset_local_float, onset_local_idx.to(dtype))  # 对第一个采样点处的 onset 关闭插值
        onset_sample = event_peak_idx[:, :, None].to(dtype) - float(self.metric_cfg.onset_search_pre) + onset_local_float  # 转换为原早期段中的绝对 onset 采样位置
        onset_sample = torch.clamp(onset_sample, min=0.0, max=float(T - 1))  # 将 onset 位置裁剪到早期段范围内
        onset_mask = has_above & event_mask[:, :, None]  # 得到每个事件每个通道的 onset 有效掩码
        onset_sample = torch.where(onset_mask, onset_sample, torch.zeros_like(onset_sample))  # 将无效 onset 位置置零

        event_energy = torch.sum(local_windows.pow(2) * local_valid[:, :, None, :].to(dtype), dim=(2, 3))  # 计算每个 impulse 局部四通道能量
        event_energy = torch.where(event_mask, event_energy, torch.zeros_like(event_energy))  # 将无效事件能量置零

        pair_i = self.pair_indices[:, 0]  # 读取 6 组通道对的第一个通道索引
        pair_j = self.pair_indices[:, 1]  # 读取 6 组通道对的第二个通道索引
        onset_i = onset_sample[:, :, pair_i]  # 取第一个通道的 onset 位置 [B, K, 6]
        onset_j = onset_sample[:, :, pair_j]  # 取第二个通道的 onset 位置 [B, K, 6]
        tdoa_pair_samples = onset_i - onset_j  # 计算 6 组 onset-TDOA，单位 sample
        tdoa_pair_seconds = tdoa_pair_samples / float(self.metric_cfg.sample_rate)  # 将 TDOA 转换为秒
        pair_mask = onset_mask[:, :, pair_i] & onset_mask[:, :, pair_j]  # 计算每组 TDOA 是否由两个有效 onset 构成
        tdoa_pair_samples = torch.where(pair_mask, tdoa_pair_samples, torch.zeros_like(tdoa_pair_samples))  # 将无效 TDOA 置零
        tdoa_pair_seconds = torch.where(pair_mask, tdoa_pair_seconds, torch.zeros_like(tdoa_pair_seconds))  # 将无效 TDOA 秒值置零

        rhs_m = -float(self.metric_cfg.sound_speed) * tdoa_pair_seconds  # 根据平面波方程构造 D u = -c tau 的右端项
        doa_raw = torch.einsum(
            'dm,bkm->bkd',
            self.pair_position_pinv.to(dtype=dtype),
            rhs_m,
        ) # 由 6 组 TDOA 最小二乘反演方向向量
        doa_norm = torch.linalg.vector_norm(doa_raw, ord=2, dim=-1, keepdim=True)  # 计算方向向量模长
        doa_to_source = doa_raw / doa_norm.clamp_min(self.metric_cfg.eps)  # 归一化得到从阵列中心指向声源/镜像声源的单位向量
        pred_rhs_m = torch.einsum('md,bkd->bkm', self.pair_position_diff.to(dtype=dtype), doa_to_source)  # 将反演 DOA 投影回 6 组 baseline
        residual_m = torch.sqrt(torch.mean((pred_rhs_m - rhs_m).pow(2), dim=-1) + self.metric_cfg.eps)  # 计算平面波拟合 RMS 残差，单位 m
        residual_samples = residual_m / float(self.metric_cfg.sound_speed) * float(self.metric_cfg.sample_rate)  # 将拟合残差转换为采样点
        all_onset_valid = onset_mask.all(dim=-1)  # 判断每个事件四个通道是否都有 onset
        valid_mask = event_mask & all_onset_valid & (doa_norm.squeeze(-1) > self.metric_cfg.eps) & (residual_samples <= float(self.metric_cfg.doa_max_residual_samples))  # 生成事件级有效掩码
        doa_to_source = torch.where(valid_mask[..., None], doa_to_source, torch.zeros_like(doa_to_source))  # 将无效事件 DOA 置零
        residual_samples = torch.where(valid_mask, residual_samples, torch.full_like(residual_samples, float('inf')))  # 将无效事件残差置为 inf

        valid_channel_count = onset_mask.to(dtype).sum(dim=-1).clamp_min(1.0)  # 统计每个事件有效通道数
        event_onset_sample = torch.sum(onset_sample * onset_mask.to(dtype), dim=-1) / valid_channel_count  # 取四通道 onset 的平均位置作为事件时间
        event_onset_sample = torch.where(event_mask, event_onset_sample, torch.zeros_like(event_onset_sample))  # 将无效事件时间置零
        event_onset_time_s = event_onset_sample / float(self.metric_cfg.sample_rate)  # 将事件时间转换为秒
        virtual_source_pos_m = float(self.metric_cfg.sound_speed) * event_onset_time_s[..., None] * doa_to_source  # 构造虚拟声源坐标 x=c*t*u
        virtual_source_pos_m = torch.where(valid_mask[..., None], virtual_source_pos_m, torch.zeros_like(virtual_source_pos_m))  # 将无效虚拟声源坐标置零

        return {  # 返回事件级分解结果字典
            'event_peak_index': event_peak_idx,  # impulse 峰值采样位置 [B, K]
            'event_mask': event_mask,  # impulse 检测有效掩码 [B, K]
            'event_valid_mask': valid_mask,  # onset + DOA 均可靠的事件掩码 [B, K]
            'event_strength': event_strength,  # impulse 联合包络强度 [B, K]
            'event_energy': event_energy,  # impulse 局部四通道能量 [B, K]
            'event_onset_sample': event_onset_sample,  # impulse 事件平均 onset 采样位置 [B, K]
            'event_onset_time_s': event_onset_time_s,  # impulse 事件平均 onset 时间 [B, K]
            'onset_sample': onset_sample,  # 四个 A-format 通道的 onset 采样位置 [B, K, 4]
            'onset_mask': onset_mask,  # 四个 A-format 通道的 onset 有效掩码 [B, K, 4]
            'tdoa_pair_samples': tdoa_pair_samples,  # 六组 onset-TDOA，单位 sample [B, K, 6]
            'tdoa_pair_seconds': tdoa_pair_seconds,  # 六组 onset-TDOA，单位秒 [B, K, 6]
            'tdoa_pair_mask': pair_mask,  # 六组 onset-TDOA 有效掩码 [B, K, 6]
            'doa_to_source': doa_to_source,  # 每个 impulse 的 DOA 单位向量 [B, K, 3]
            'virtual_source_pos_m': virtual_source_pos_m,  # 每个 impulse 的虚拟声源坐标 [B, K, 3]
            'plane_wave_residual_samples': residual_samples,  # 平面波拟合残差，单位 sample [B, K]
        }

    # --------------------------------------------------------
    # DPOT-SDM 距离与事件级误差
    # --------------------------------------------------------
    def _sinkhorn_transport(self, cost: torch.Tensor, source_mass: torch.Tensor, target_mass: torch.Tensor) -> torch.Tensor:
        """对 batched cost matrix 执行熵正则 Sinkhorn，返回 transport plan。"""
        reg = max(float(self.metric_cfg.dpot_sinkhorn_reg_m), self.metric_cfg.eps)  # 读取并保护正则系数
        kernel = torch.exp(-cost / reg).clamp_min(self.metric_cfg.eps)  # 将代价矩阵转换为 Gibbs kernel
        u = torch.ones_like(source_mass)  # 初始化左缩放向量
        v = torch.ones_like(target_mass)  # 初始化右缩放向量
        for _ in range(int(self.metric_cfg.dpot_sinkhorn_iters)):  # 执行固定次数 Sinkhorn 迭代
            v = target_mass / (torch.matmul(kernel.transpose(1, 2), u.unsqueeze(-1)).squeeze(-1) + self.metric_cfg.eps)  # 更新右缩放向量
            u = source_mass / (torch.matmul(kernel, v.unsqueeze(-1)).squeeze(-1) + self.metric_cfg.eps)  # 更新左缩放向量
        transport = u[:, :, None] * kernel * v[:, None, :]  # 组合得到 transport plan [B, N, M]
        return transport  # 返回传输矩阵

    def dpot_sdm_distance_and_errors(self, target_events: Dict[str, torch.Tensor], prediction_events: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        基于 dustbin partial OT 比较两组 impulse-wise SDM 事件集合。\n
        返回：\n
        1. DPOT-SDM 距离，单位 m；\n
        2. matched onset-TDOA 平均绝对误差，单位 ms；\n
        3. matched onset-DOA 角误差，单位 degree；\n
        4. 有效匹配掩码；\n
        5. 真实事件数和预测事件数的平均数量误差。\n
        """
        target_pos = target_events['virtual_source_pos_m']  # 读取目标虚拟声源坐标 [B, K, 3]
        prediction_pos = prediction_events['virtual_source_pos_m']  # 读取预测虚拟声源坐标 [B, K, 3]
        target_valid = target_events['event_valid_mask']  # 读取目标有效事件掩码 [B, K]
        prediction_valid = prediction_events['event_valid_mask']  # 读取预测有效事件掩码 [B, K]
        target_energy = target_events['event_energy']  # 读取目标事件能量 [B, K]
        prediction_energy = prediction_events['event_energy']  # 读取预测事件能量 [B, K]

        B, K, _ = target_pos.shape  # 读取 batch 和事件数量上限
        target_mass_raw = self.event_mass_from_energy(target_energy, target_valid)  # 根据配置计算目标事件质量
        prediction_mass_raw = self.event_mass_from_energy(prediction_energy, prediction_valid)  # 根据配置计算预测事件质量
        target_real_sum = target_mass_raw.sum(dim=-1, keepdim=True)  # 计算目标事件总质量
        prediction_real_sum = prediction_mass_raw.sum(dim=-1, keepdim=True)  # 计算预测事件总质量
        target_has_event = target_real_sum.squeeze(-1) > self.metric_cfg.eps  # 判断目标是否存在有效事件
        prediction_has_event = prediction_real_sum.squeeze(-1) > self.metric_cfg.eps  # 判断预测是否存在有效事件
        target_mass = torch.where(target_has_event[:, None], target_mass_raw / target_real_sum.clamp_min(self.metric_cfg.eps), target_mass_raw)  # 将目标事件质量归一化到总和 1
        prediction_mass = torch.where(prediction_has_event[:, None], prediction_mass_raw / prediction_real_sum.clamp_min(self.metric_cfg.eps), prediction_mass_raw)  # 将预测事件质量归一化到总和 1

        cost_real = torch.cdist(target_pos, prediction_pos, p=2.0)  # 计算真实事件之间的虚拟声源欧氏距离，单位 m
        valid_pair = target_valid[:, :, None] & prediction_valid[:, None, :]  # 生成真实事件对有效掩码
        cost_real = torch.where(valid_pair, cost_real, torch.full_like(cost_real, float(self.metric_cfg.dpot_unmatched_cost_m)))  # 对无效事件对设置 dustbin 等价高代价

        N_aug = K + 1  # 增加一个 dustbin 后的事件数量
        cost_aug = torch.zeros(B, N_aug, N_aug, device=target_pos.device, dtype=target_pos.dtype)  # 初始化增强代价矩阵
        cost_aug[:, :K, :K] = cost_real  # 写入真实事件之间的代价
        cost_aug[:, :K, K] = float(self.metric_cfg.dpot_unmatched_cost_m)  # 目标真实事件到预测 dustbin 的未匹配代价
        cost_aug[:, K, :K] = float(self.metric_cfg.dpot_unmatched_cost_m)  # 预测真实事件到目标 dustbin 的未匹配代价
        cost_aug[:, K, K] = 0.0  # dustbin 到 dustbin 的代价为零

        source_mass = torch.zeros(B, N_aug, device=target_pos.device, dtype=target_pos.dtype)  # 初始化源分布质量
        target_mass_aug = torch.zeros(B, N_aug, device=target_pos.device, dtype=target_pos.dtype)  # 初始化目标分布质量
        source_mass[:, :K] = target_mass  # 源分布真实质量来自目标 SRIR 事件
        target_mass_aug[:, :K] = prediction_mass  # 目标分布真实质量来自预测 SRIR 事件
        source_mass[:, K] = float(self.metric_cfg.dpot_dummy_mass) + prediction_has_event.to(dtype=target_pos.dtype)  # 源 dustbin 容纳预测未匹配事件
        target_mass_aug[:, K] = float(self.metric_cfg.dpot_dummy_mass) + target_has_event.to(dtype=target_pos.dtype)  # 目标 dustbin 容纳目标未匹配事件

        transport_aug = self._sinkhorn_transport(cost_aug, source_mass, target_mass_aug)  # 计算增强 partial OT transport plan
        transport_real = transport_aug[:, :K, :K]  # 取真实事件之间的传输子矩阵
        dpot_distance_m = torch.sum(transport_aug * cost_aug, dim=(1, 2))  # 计算 DPOT-SDM 距离，单位 m

        target_tdoa = target_events['tdoa_pair_seconds']  # 读取目标事件 6 组 TDOA，单位 s
        prediction_tdoa = prediction_events['tdoa_pair_seconds']  # 读取预测事件 6 组 TDOA，单位 s
        pair_tdoa_err_ms = torch.mean(torch.abs(target_tdoa[:, :, None, :] - prediction_tdoa[:, None, :, :]), dim=-1) * 1000.0  # 计算事件对之间的平均 TDOA 误差，单位 ms

        target_doa = target_events['doa_to_source']  # 读取目标事件 DOA [B, K, 3]
        prediction_doa = prediction_events['doa_to_source']  # 读取预测事件 DOA [B, K, 3]
        doa_dot = torch.sum(target_doa[:, :, None, :] * prediction_doa[:, None, :, :], dim=-1)  # 计算事件对 DOA 点积
        doa_dot = torch.clamp(doa_dot, min=-1.0, max=1.0)  # 将点积裁剪到 arccos 合法范围
        pair_doa_err_deg = torch.rad2deg(torch.acos(doa_dot))  # 将角距离转换为度
        pair_doa_err_deg = torch.where(valid_pair, pair_doa_err_deg, torch.zeros_like(pair_doa_err_deg))  # 将无效事件对角误差置零
        pair_tdoa_err_ms = torch.where(valid_pair, pair_tdoa_err_ms, torch.zeros_like(pair_tdoa_err_ms))  # 将无效事件对 TDOA 误差置零

        matched_mass = torch.sum(transport_real, dim=(1, 2))  # 计算真实事件之间的总匹配质量
        matched_valid = matched_mass > float(self.metric_cfg.matched_mass_min)  # 判断是否存在足够真实匹配质量
        onset_tdoa_err_ms = torch.sum(transport_real * pair_tdoa_err_ms, dim=(1, 2)) / matched_mass.clamp_min(self.metric_cfg.eps)  # 用 transport 对 TDOA 误差加权平均
        onset_doa_err_deg = torch.sum(transport_real * pair_doa_err_deg, dim=(1, 2)) / matched_mass.clamp_min(self.metric_cfg.eps)  # 用 transport 对 DOA 误差加权平均
        onset_tdoa_err_ms = torch.where(matched_valid, onset_tdoa_err_ms, torch.zeros_like(onset_tdoa_err_ms))  # 无匹配时将 TDOA 误差置零
        onset_doa_err_deg = torch.where(matched_valid, onset_doa_err_deg, torch.zeros_like(onset_doa_err_deg))  # 无匹配时将 DOA 误差置零

        target_count = target_valid.to(dtype=target_pos.dtype).sum(dim=-1)  # 统计目标有效事件数量
        prediction_count = prediction_valid.to(dtype=target_pos.dtype).sum(dim=-1)  # 统计预测有效事件数量
        event_count_abs_err = torch.abs(target_count - prediction_count)  # 计算有效事件数量绝对误差

        return dpot_distance_m, onset_tdoa_err_ms, onset_doa_err_deg, matched_valid, event_count_abs_err  # 返回 DPOT 与事件级误差

    # --------------------------------------------------------
    # IACC 指标
    # --------------------------------------------------------
    def _iacc_matrix(self, x: torch.Tensor) -> torch.Tensor:
        """计算四通道 A-format/FOA SRIR 的类 IACC 最大互相关矩阵，输出 [B, 4, 4]。"""
        if x.ndim != 3 or x.shape[1] != 4:  # 检查输入形状
            raise ValueError('IACC matrix estimation requires input shape [B, 4, T].')  # 抛出维度错误

        x = x.float()  # 将输入统一为 float32
        B, C, T = x.shape  # 读取 batch、通道数和时间长度
        n_fft = self._next_power_of_two(2 * T - 1)  # 选择足够长的 FFT 以获得线性互相关
        spec = torch.fft.rfft(x, n=n_fft, dim=-1)  # 对四通道信号做实数 FFT
        energy = torch.sum(x ** 2, dim=-1).clamp_min(self.metric_cfg.eps)  # 计算每个通道能量并避免除零
        iacc_matrix = torch.eye(C, device=x.device, dtype=x.dtype).unsqueeze(0).repeat(B, 1, 1)  # 初始化 IACC 矩阵，对角线为 1
        max_lag = min(self.iacc_lag_samples, max(T - 1, 0))  # 确定实际可用的最大滞后点数

        for mic_i, mic_j in self.pair_indices.tolist():  # 遍历四通道的全部 6 组 pair
            cross_spec = spec[:, mic_i, :] * torch.conj(spec[:, mic_j, :])  # 构造互功率谱
            corr_full = torch.fft.irfft(cross_spec, n=n_fft, dim=-1)  # 逆 FFT 得到循环互相关
            if max_lag > 0:  # 若允许非零滞后搜索
                corr_lag = torch.cat([corr_full[:, -max_lag:], corr_full[:, :max_lag + 1]], dim=-1)  # 拼接负滞后和非负滞后区间
            else:  # 若只允许零滞后
                corr_lag = corr_full[:, :1]  # 仅保留零滞后互相关
            denominator = torch.sqrt(energy[:, mic_i] * energy[:, mic_j] + self.metric_cfg.eps).unsqueeze(-1)  # 构造归一化分母
            norm_corr = corr_lag / denominator  # 得到归一化互相关
            max_corr = torch.max(torch.abs(norm_corr), dim=-1).values  # 在 ±lag 内取最大绝对相关值
            max_corr = torch.clamp(max_corr, min=0.0, max=1.0 + 1e-6)  # 将 IACC 限制在合理范围内
            iacc_matrix[:, mic_i, mic_j] = max_corr  # 写入上三角/下三角中的一个位置
            iacc_matrix[:, mic_j, mic_i] = max_corr  # 写入对称位置
        return iacc_matrix  # 返回 [B, 4, 4] IACC 矩阵

    def iacc_matrix_relative_error(self, target: torch.Tensor, prediction: torch.Tensor) -> torch.Tensor:
        """计算四通道 IACC 矩阵相对 Frobenius 误差，返回 [B]。"""
        target_matrix = self._iacc_matrix(target)  # 计算目标 SRIR 的 IACC 矩阵
        prediction_matrix = self._iacc_matrix(prediction)  # 计算预测 SRIR 的 IACC 矩阵
        matrix_diff = target_matrix - prediction_matrix  # 计算 IACC 矩阵差
        diff_norm = torch.linalg.vector_norm(matrix_diff.reshape(matrix_diff.shape[0], -1), ord=2, dim=-1)  # 计算差矩阵 Frobenius 范数
        target_norm = torch.linalg.vector_norm(target_matrix.reshape(target_matrix.shape[0], -1), ord=2, dim=-1).clamp_min(self.metric_cfg.eps)  # 计算目标矩阵范数
        return diff_norm / target_norm  # 返回相对 Frobenius 误差

    # --------------------------------------------------------
    # batch 级指标接口
    # --------------------------------------------------------
    def compute_batch_metrics(self, target: torch.Tensor, prediction: torch.Tensor, compute_full_metrics: bool = True) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """计算一个 batch 的逐样本空间事件指标，并返回逐指标聚合权重。"""
        early_len = min(self.metric_cfg.early_len, target.shape[-1], prediction.shape[-1])  # 计算当前 batch 可用的早期段长度
        target_early = target[:, :, :early_len]  # 截取目标 SRIR 前 4096 点
        prediction_early = prediction[:, :, :early_len]  # 截取预测 SRIR 前 4096 点

        target_events = self.decompose_impulse_sdm(target_early)  # 分解目标 SRIR 的早期 impulse-wise SDM 事件
        prediction_events = self.decompose_impulse_sdm(prediction_early)  # 分解预测 SRIR 的早期 impulse-wise SDM 事件
        dpot_distance_m, onset_tdoa_err_ms, onset_doa_err_deg, matched_valid, event_count_abs_err = self.dpot_sdm_distance_and_errors(target_events, prediction_events)  # 计算 DPOT-SDM 及匹配事件误差

        iacc_4096_err = self.iacc_matrix_relative_error(target_early, prediction_early)  # 计算前 4096 点 IACC 矩阵相对误差
        target_event_count = target_events['event_valid_mask'].to(dtype=target.dtype).sum(dim=-1)  # 统计目标有效 impulse 数量
        prediction_event_count = prediction_events['event_valid_mask'].to(dtype=target.dtype).sum(dim=-1)  # 统计预测有效 impulse 数量
        one_weight = torch.ones(target.shape[0], device=target.device, dtype=torch.float32)  # 构造等权权重
        early_weight = self.aggregation_weight(target_early)  # 根据配置计算早期段聚合权重
        matched_weight = early_weight * matched_valid.to(dtype=early_weight.dtype)  # 对需要真实匹配的误差使用 matched_valid 屏蔽无效样本

        metric_dic = {  # 构造逐样本指标字典
            'Onset_TDOA_event_DPOT_err_ms': onset_tdoa_err_ms,  # DPOT 匹配后的事件级 onset-TDOA 误差
            'Onset_DOA_event_DPOT_angular_err_deg': onset_doa_err_deg,  # DPOT 匹配后的事件级 onset-DOA 角误差
            'DPOT_SDM_4096_distance_m': dpot_distance_m,  # 前 4096 点事件点云的 DPOT-SDM 距离
            'IACC_matrix_4096_rel_err': iacc_4096_err,  # 前 4096 点四通道 IACC 矩阵相对误差
            'Onset_SDM_matched_valid_ratio': matched_valid.to(dtype=target.dtype),  # onset/SDM 匹配有效比例
            'Target_impulse_count_4096': target_event_count,  # 目标前 4096 点有效 impulse 数量
            'Prediction_impulse_count_4096': prediction_event_count,  # 预测前 4096 点有效 impulse 数量
            'Impulse_count_4096_abs_err': event_count_abs_err,  # 前 4096 点有效 impulse 数量绝对误差
        }

        weight_dic = {  # 构造每个指标对应的数据集级聚合权重
            'Onset_TDOA_event_DPOT_err_ms': matched_weight,  # TDOA 误差仅在存在有效匹配时参与聚合
            'Onset_DOA_event_DPOT_angular_err_deg': matched_weight,  # DOA 误差仅在存在有效匹配时参与聚合
            'DPOT_SDM_4096_distance_m': early_weight,  # DPOT 距离使用配置指定的早期段能量权重
            'IACC_matrix_4096_rel_err': early_weight,  # 4096 点 IACC 误差使用配置指定的早期段能量权重
            'Onset_SDM_matched_valid_ratio': one_weight,  # 有效比例使用逐样本等权
            'Target_impulse_count_4096': one_weight,  # 事件数量使用逐样本等权
            'Prediction_impulse_count_4096': one_weight,  # 事件数量使用逐样本等权
            'Impulse_count_4096_abs_err': one_weight,  # 数量误差使用逐样本等权
        }

        if compute_full_metrics:  # 若需要计算完整信号指标
            iacc_full_err = self.iacc_matrix_relative_error(target, prediction)  # 计算完整 SRIR 的 IACC 矩阵相对误差
            metric_dic['IACC_matrix_full_rel_err'] = iacc_full_err  # 写入完整信号 IACC 误差
            weight_dic['IACC_matrix_full_rel_err'] = self.aggregation_weight(target)  # 完整信号 IACC 使用完整目标信号权重

        return metric_dic, weight_dic  # 返回逐样本指标和聚合权重


