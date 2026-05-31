from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from time import time  # 用于计时等操作
import torch
import torch.nn as nn
import torch.nn.parallel  # 并行计算相关
from CConfig import config  # 导入自定义配置模块
cfg = config()  # 实例化配置对象

import numpy as np
import torch_geometric.nn as pygn  # 导入 PyTorch Geometric 的神经网络模块




class PositionalEnc(nn.Module):
    def __init__(self, n_pos_repeat=8, pos_dim=9):
        """
        位置编码模块：将低维的连续位置特征（如 xyz pitch yaw 等）编码为高维向量，
        便于网络学习空间关系，类似于 Transformer 中的位置编码。
        """
        super().__init__()
        self.MIN_FREQ = 1e-2  # 最小频率，用于控制编码的频谱范围
        self._n_positional_obs = pos_dim  # 输入的位置维度数量（如 xyz + 方向角等）
        self.n_pos_repeat = n_pos_repeat
        self.out_dim = (self.n_pos_repeat + 1) * self._n_positional_obs
        # 构造频率向量：模仿 Transformer 的 sinusoidal encoding
        self._freqs = self.MIN_FREQ ** (
            2 * (torch.div(torch.arange(self.n_pos_repeat, dtype=torch.float32),
                           2, rounding_mode='trunc')) / self.n_pos_repeat
        )

    def forward(self, positional_obs):
        """
        :param positional_obs: 形状 (batch_size, 6) 的位置张量
        :return: 编码后的高维位置向量，形状为 (batch_size, 54)
        """
        positional_enc_out = [positional_obs]  # 初始输出包含原始输入

        assert len(positional_obs.size()) == 2
        assert positional_obs.size(-1) == self._n_positional_obs  # 检查维度合法性

        # 将频率向量扩展成 batch 维度
        freqs = self._freqs.unsqueeze(0).repeat((positional_obs.size(0), 1)).to(positional_obs.device)

        # 对每个位置维度做编码
        for positional_obs_idx in range(self._n_positional_obs):
            positional_obs_this_idx = positional_obs[:, positional_obs_idx].unsqueeze(-1)  # 取出当前维度，(B, 1)
            positional_obs_this_idx = positional_obs_this_idx * freqs  # 频率调制

            positional_obs_this_idx_clone = positional_obs_this_idx.clone()
            positional_obs_this_idx_clone[..., ::2] = (positional_obs[:, positional_obs_idx].unsqueeze(-1) *
                                                       torch.cos(positional_obs_this_idx[..., ::2]))  # 偶数维cos
            positional_obs_this_idx_clone[..., 1::2] = (positional_obs[:, positional_obs_idx].unsqueeze(-1) *
                                                        torch.sin(positional_obs_this_idx[..., 1::2]))  # 奇数维sin

            positional_enc_out.append(positional_obs_this_idx_clone)  # 添加到输出中

        positional_enc_out = torch.cat(positional_enc_out, dim=-1)  # 拼接所有维度编码
        return positional_enc_out




def upBlock(in_planes, out_planes, kernel_length, stride, BN=False, NL='prelu', padding=None):
    """上采样模块：stride=2，同样是反卷积 + BN + 激活"""
    if stride > 1:
        output_padding = 1
    else:
        output_padding = 0
    if padding is None:
        padding = (kernel_length + 1 - stride) // 2
    layers = [nn.ConvTranspose1d(in_planes, out_planes,
                           kernel_size=kernel_length, stride=stride,
                           padding=padding, output_padding=output_padding)]
    if BN:
        layers.append(nn.BatchNorm1d(out_planes))
    if NL=='prelu':
        layers.append(nn.PReLU())
    elif NL=='sigmoid':
        layers.append(nn.Sigmoid())
    elif NL=='tanh':
        layers.append(nn.Tanh())
    elif NL=='relu':
        layers.append(nn.ReLU())
    elif NL == 'none':
        layers.append(nn.Identity())
    return nn.Sequential(*layers)




def Conv1Block(in_planes, out_planes, kernel_length, BN=False, NL='prelu', padding=None, stride=None):
    """上采样模块：stride=2，同样是反卷积 + BN + 激活"""
    if padding is None:
        padding = (kernel_length - 1) // 2
    if stride is None:
        stride = 1
    layers = [nn.ConvTranspose1d(in_planes, out_planes, kernel_size=kernel_length,
                           padding=padding, stride=stride)]
    if BN:
        layers.append(nn.BatchNorm1d(out_planes))
    if NL=='prelu':
        layers.append(nn.PReLU())
    elif NL=='sigmoid':
        layers.append(nn.Sigmoid())
    elif NL=='tanh':
        layers.append(nn.Tanh())
    elif NL=='relu':
        layers.append(nn.ReLU())
    elif NL=='none':
        layers.append(nn.Identity())
    return nn.Sequential(*layers)




def DownBlock(in_planes, out_planes, kernel_length, stride, BN=False, NL='prelu'):
    """下采样模块：stride=2，同样是反卷积 + BN + 激活"""
    layers = [nn.Conv1d(in_planes, out_planes, kernel_size=kernel_length,
                           padding=(kernel_length - stride) // 2, stride=stride)]
    if BN:
        layers.append(nn.BatchNorm1d(out_planes))
    if NL=='prelu':
        layers.append(nn.PReLU())
    elif NL=='sigmoid':
        layers.append(nn.Sigmoid())
    elif NL=='tanh':
        layers.append(nn.Tanh())
    elif NL=='relu':
        layers.append(nn.ReLU())
    elif NL=='none':
        layers.append(nn.Identity())
    return nn.Sequential(*layers)




def fcBlock(in_planes, out_planes, bias=True, BN=True, nonlin_ly='prelu'):
    """
    构造一个包含 Linear + 可选 BN + 激活 的 MLP block。
    nonlin_ly 可选: 'prelu', 'relu', 'tanh', 'sigmoid'，也可扩展
    """
    layers = [nn.Linear(in_planes, out_planes, bias=bias)]
    if BN:
        layers.append(nn.BatchNorm1d(out_planes))
    if nonlin_ly=='prelu':
        layers.append(nn.PReLU())
    elif nonlin_ly=='sigmoid':
        layers.append(nn.Sigmoid())
    elif nonlin_ly=='tanh':
        layers.append(nn.Tanh())
    elif nonlin_ly=='relu':
        layers.append(nn.ReLU())
    return nn.Sequential(*layers)




class Transformer_Decoder(nn.Module):
    def __init__(self, d_model, nhead, num_layers, dim_feedforward,
                 batch_first=True, dropout=0.1):
        """
        d_model: 输入和输出特征的维度（即嵌入维度）。
        nhead: 多头注意力的头数。
        num_layers: 解码器层的层数。
        dim_feedforward: 前馈神经网络的隐藏层维度。
        batch_first: 是否将 batch size 作为输入的第一个维度。默认为True
        dropout: dropout 概率，用于防止过拟合。默认为0.1
        """
        super(Transformer_Decoder, self).__init__()
        self.TfDcd_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            batch_first=batch_first, dropout=dropout
        )
        self.TfDcd = nn.TransformerDecoder(self.TfDcd_layer, num_layers=num_layers)

    def forward(self, query_seq, memory_seq):
        """
        query_seq: 解码器的输入序列（通常是目标序列的嵌入）。
        memory_seq: 来自编码器的输出序列（即上下文记忆）。
        """
        query_seq = self.TfDcd(query_seq, memory_seq)
        return query_seq




class POS_ENC(nn.Module):
    def __init__(self, feature_dim_in, feature_dim_out):
        super(POS_ENC, self).__init__()
        self.pos_sin_enc = PositionalEnc(pos_dim=6)
        self.pos_fc1 = fcBlock(in_planes=feature_dim_in, out_planes=feature_dim_out // 2,
                               bias=False, BN=False, nonlin_ly='prelu')
        self.pos_fc2 = fcBlock(in_planes=feature_dim_out // 2, out_planes=feature_dim_out,
                               bias=False, BN=False, nonlin_ly='prelu')


    def forward(self, pos):
        pos = self.pos_sin_enc(pos)
        pos = self.pos_fc1(pos)
        pos = self.pos_fc2(pos)
        return pos




if __name__ == '__main__':
    lst_src_pos = torch.randn(2, 6) * 10
    Pos_Enc = PositionalEnc(16)
    pe = Pos_Enc(lst_src_pos)
    pe_np = (pe.detach().numpy())[0]