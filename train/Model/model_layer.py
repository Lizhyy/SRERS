from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
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
    def __init__(self, n_pos_repeat=8):
        """
        位置编码模块：将低维的连续位置特征（如 xyz pitch yaw 等）编码为高维向量，
        便于网络学习空间关系，类似于 Transformer 中的位置编码。
        """
        super().__init__()
        self.MIN_FREQ = 1e-2  # 最小频率，用于控制编码的频谱范围
        self._n_positional_obs = 6  # 输入的位置维度数量（如 xyz + 方向角等）
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




def old_conv3x1(in_planes, out_planes, stride=1):
    """1D 卷积模块，核宽为3，padding=1，使输出长度与输入相近"""
    kernel_size = 3
    return nn.Conv1d(in_planes, out_planes,
                     kernel_size=kernel_size, stride=stride,
                     padding=1, bias=False)



def upBlock4(in_planes, out_planes):
    """上采样模块：stride=4，带 BN 和 PReLU 激活"""
    kernel_length = 41
    stride = 4
    block = nn.Sequential(
        nn.ConvTranspose1d(in_planes, out_planes,
                           kernel_size=kernel_length, stride=stride,
                           padding=19, output_padding=1),
        nn.BatchNorm1d(out_planes),
        nn.PReLU())
    return block




def upBlock2(in_planes, out_planes):
    """上采样模块：stride=2，同样是反卷积 + BN + 激活"""
    kernel_length = 41
    stride = 2
    block = nn.Sequential(
        nn.ConvTranspose1d(in_planes, out_planes,
                           kernel_size=kernel_length, stride=stride,
                           padding=20, output_padding=1),
        nn.BatchNorm1d(out_planes),
        nn.PReLU())
    return block



def upBlock1(in_planes, out_planes):
    """上采样模块：stride=2，同样是反卷积 + BN + 激活"""
    kernel_length = 41
    stride = 1
    block = nn.Sequential(
        nn.ConvTranspose1d(in_planes, out_planes,
                           kernel_size=kernel_length, stride=stride,
                           padding=20)
        )
    return block



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



def Conv2_same(in_channels, out_channels, kernel_size, BN=False, NL='prelu', stride=1):
    """上采样模块：stride=2，同样是反卷积 + BN + 激活"""
    ph, pw = same_padding(kernel_size, stride, 1)
    padding = (ph, pw)

    layers = [
        nn.Conv2d(in_channels, out_channels,
                  kernel_size=kernel_size,
                  stride=stride,
                  padding=padding,
                  bias=False)
    ]
    if BN:
        layers.append(nn.BatchNorm2d(out_channels))
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






class CovTrans2ResBlock2(nn.Module):
    def __init__(self, in_planes, out_planes, Ts_kernel, kernel1, kernel2, kernel3):
        super(CovTrans2ResBlock2, self).__init__()
        self.ConvTrans = upBlock(in_planes, out_planes, stride=2, kernel_length=Ts_kernel,
                                  BN=True, NL='prelu', output_padding=1,
                                  padding=(Ts_kernel - 1) // 2)
        self.Conv1 = Conv1Block(out_planes, out_planes,
                                BN=True, NL='prelu', kernel_length=kernel1)
        self.ConvSum1 = Conv1Block(out_planes * 2, out_planes,
                                  BN=True, NL='prelu', kernel_length=1)
        self.Conv2 = Conv1Block(out_planes, out_planes,
                                BN=True, NL='prelu', kernel_length=kernel2)
        self.ConvSum2 = Conv1Block(out_planes * 2, out_planes,
                                  BN=True, NL='prelu', kernel_length=1)
        self.Conv3 = Conv1Block(out_planes, out_planes,
                                BN=True, NL='prelu', kernel_length=kernel3)
        self.ConvSum3 = Conv1Block(out_planes * 2, out_planes,
                                  BN=True, NL='prelu', kernel_length=1)

    def forward(self, RIR_wav):
        RIR_wav = self.ConvTrans(RIR_wav)
        RIR_wav1 = self.Conv1(RIR_wav)
        RIR_wav = torch.cat((RIR_wav, RIR_wav1), dim=1)
        RIR_wav = self.ConvSum1(RIR_wav)

        RIR_wav1 = self.Conv2(RIR_wav)
        RIR_wav = torch.cat((RIR_wav, RIR_wav1), dim=1)
        RIR_wav = self.ConvSum2(RIR_wav)

        RIR_wav1 = self.Conv3(RIR_wav)
        RIR_wav = torch.cat((RIR_wav, RIR_wav1), dim=1)
        RIR_wav = self.ConvSum3(RIR_wav)
        return RIR_wav




class ConvResBlock(nn.Module):
    def __init__(self, in_planes, out_planes, Ts_kernel, kernel1, kernel2, kernel3):
        super(ConvResBlock, self).__init__()
        self.ConvTrans = upBlock(in_planes, out_planes, stride=2, kernel_length=Ts_kernel,
                                  BN=True, NL='prelu', output_padding=1,
                                  padding=(Ts_kernel - 1) // 2)
        self.Conv1 = Conv1Block(out_planes, out_planes,
                                BN=True, NL='prelu', kernel_length=kernel1)
        self.Conv2 = Conv1Block(out_planes, out_planes,
                                BN=True, NL='prelu', kernel_length=kernel2)
        self.Conv3 = Conv1Block(out_planes, out_planes,
                                BN=True, NL='prelu', kernel_length=kernel3)
        self.ConvSum = Conv1Block(out_planes * 4, out_planes,
                                  BN=True, NL='prelu', kernel_length=1)

    def forward(self, RIR_wav):
        RIR_wav = self.ConvTrans(RIR_wav)
        RIR_wav1 = self.Conv1(RIR_wav)
        RIR_wav2 = self.Conv2(RIR_wav)
        RIR_wav3 = self.Conv3(RIR_wav)
        RIR_wav = torch.cat((RIR_wav, RIR_wav1, RIR_wav2, RIR_wav3), dim=1)
        RIR_wav = self.ConvSum(RIR_wav)
        return RIR_wav




class RIR_out_ResBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel1, kernel2, kernel3, kernel4):
        super(RIR_out_ResBlock, self).__init__()
        self.Conv1 = Conv1Block(in_planes, in_planes,
                                BN=True, NL='prelu', kernel_length=kernel1)
        self.Conv2 = Conv1Block(in_planes, in_planes,
                                BN=True, NL='prelu', kernel_length=kernel2)
        self.Conv3 = Conv1Block(in_planes, in_planes,
                                BN=True, NL='prelu', kernel_length=kernel3)
        self.Conv4 = Conv1Block(in_planes, in_planes,
                                BN=True, NL='prelu', kernel_length=kernel4)
        self.ConvSum = Conv1Block(in_planes * 4, out_planes,
                                  BN=False, NL='tanh', kernel_length=1)

    def forward(self, RIR_wav):
        RIR_wav1 = self.Conv1(RIR_wav)
        RIR_wav2 = self.Conv2(RIR_wav)
        RIR_wav3 = self.Conv3(RIR_wav)
        RIR_wav4 = self.Conv4(RIR_wav)
        RIR_wav = torch.cat((RIR_wav1, RIR_wav2, RIR_wav3, RIR_wav4), dim=1)
        RIR_wav = self.ConvSum(RIR_wav)
        return RIR_wav




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



class Transformer_Encoder(nn.Module):
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
        super(Transformer_Encoder, self).__init__()
        self.TfEcd_layer = nn.TransformerEncoderLayer(d_model=d_model,
                                                      nhead=nhead,
                                                      dim_feedforward=dim_feedforward,
                                                      batch_first=batch_first,
                                                      dropout=dropout
                                                      )
        self.TfEcd = nn.TransformerEncoder(self.TfEcd_layer, num_layers=num_layers)

    def forward(self, in_seq):
        """
        in_seq: 形状通常为 (batch_size, seq_len, d_model)（若 batch_first=True）。
        """
        in_seq = self.TfEcd(in_seq)
        return in_seq



class Mesh_En_layer(nn.Module):
    def __init__(self, feature_dim_in, feature_dim_out, ratio,
                 GCN_layer='GAT', Gpool_layer='TopK', NL='prelu'):
        """
        图神经网络编码层：支持 GCN / GAT 图卷积，支持 TopK / SAG 图池化
        """
        super(Mesh_En_layer, self).__init__()
        if NL == 'relu':
            self.prelu = nn.ReLU()
        elif NL == 'tanh':
            self.prelu = nn.Tanh()
        else:
            self.prelu = nn.PReLU()

        # 图卷积选择
        if GCN_layer == 'GCN':
            self.graph_conv = pygn.GCNConv(in_channels=feature_dim_in,
                                           out_channels=feature_dim_out)
        elif GCN_layer == 'GAT':
            self.graph_conv = pygn.GATv2Conv(in_channels=feature_dim_in,
                                             out_channels=feature_dim_out)

        # 图池化选择
        if Gpool_layer == 'TopK':
            self.gpool = pygn.TopKPooling(in_channels=feature_dim_out, ratio=ratio)
        elif Gpool_layer == 'SAG':
            self.gpool = pygn.SAGPooling(in_channels=feature_dim_out, ratio=ratio)

    def forward(self, x, edge_index, batch):
        """
        :param x: 节点特征 (num_nodes, F)
        :param edge_index: 边索引 (2, num_edges)
        :param batch: 图的 batch 标签 (num_nodes,)
        :return: 池化后的节点特征、边索引和 batch 标签
        """
        x = self.prelu(self.graph_conv(x=x, edge_index=edge_index))  # 图卷积 + 激活
        x, edge_index, _, batch, _, _ = self.gpool(x=x, edge_index=edge_index,
                                                   batch=batch)  # 图池化
        return x, edge_index, batch



class Mesh_En_ddim_res(nn.Module):
    def __init__(self, feature_dim_in, feature_dim_out, ratio1, ratio2,
                 GCN_layer='GAT', Gpool_layer='TopK'):
        """
        图神经网络编码层：支持 GCN / GAT 图卷积，支持 TopK / SAG 图池化
        """
        super(Mesh_En_ddim_res, self).__init__()
        self.prelu = nn.PReLU()
        self.feature_dim_out = feature_dim_out
        self.ratio2 = ratio2

        # 图卷积选择
        if GCN_layer == 'GCN':
            self.graph_conv1 = pygn.GCNConv(in_channels=feature_dim_in,
                                            out_channels=feature_dim_out)
            self.graph_conv2 = pygn.GCNConv(in_channels=feature_dim_out,
                                            out_channels=feature_dim_out)
        elif GCN_layer == 'GAT':
            self.graph_conv1 = pygn.GATv2Conv(in_channels=feature_dim_in,
                                              out_channels=feature_dim_out)
            self.graph_conv2 = pygn.GATv2Conv(in_channels=feature_dim_out,
                                              out_channels=feature_dim_out)

        # 图池化选择
        if Gpool_layer == 'TopK':
            self.gpool1 = pygn.TopKPooling(in_channels=feature_dim_out, ratio=ratio1)
            self.gpool2 = pygn.TopKPooling(in_channels=feature_dim_out, ratio=ratio2)
        elif Gpool_layer == 'SAG':
            self.gpool1 = pygn.SAGPooling(in_channels=feature_dim_out, ratio=ratio1)
            self.gpool2 = pygn.SAGPooling(in_channels=feature_dim_out, ratio=ratio2)

    def forward(self, x, edge_index, batch):
        """
        :param x: 节点特征 (num_nodes, F)
        :param edge_index: 边索引 (2, num_edges)
        :param batch: 图的 batch 标签 (num_nodes,)
        :return: 池化后的节点特征、边索引和 batch 标签
        """
        x = self.prelu(self.graph_conv1(x=x, edge_index=edge_index))  # 图卷积 + 激活
        x1 = self.prelu(self.graph_conv2(x=x, edge_index=edge_index))
        x, edge_index, _, batch, _, _ = self.gpool1(x=(x + x1), edge_index=edge_index,
                                                     batch=batch)  # 图池化
        x1, _, _, batch2, _, _ = self.gpool2(x=x, edge_index=edge_index, batch=batch)  # 图池化
        x1 = x1.view(torch.max(batch2) + 1, self.ratio2, self.feature_dim_out)
        return x, x1, edge_index, batch




class Mesh_En_ddim(nn.Module):
    def __init__(self, feature_dim_in, feature_dim_out, ratio1, ratio2,
                 GCN_layer='GAT', Gpool_layer='TopK'):
        """
        图神经网络编码层：支持 GCN / GAT 图卷积，支持 TopK / SAG 图池化
        """
        super(Mesh_En_ddim, self).__init__()
        self.prelu = nn.PReLU()
        self.feature_dim_out = feature_dim_out
        self.ratio2 = ratio2

        # 图卷积选择
        if GCN_layer == 'GCN':
            self.graph_conv1 = pygn.GCNConv(in_channels=feature_dim_in,
                                            out_channels=feature_dim_out)
        elif GCN_layer == 'GAT':
            self.graph_conv1 = pygn.GATv2Conv(in_channels=feature_dim_in,
                                              out_channels=feature_dim_out)

        # 图池化选择
        if Gpool_layer == 'TopK':
            self.gpool1 = pygn.TopKPooling(in_channels=feature_dim_out, ratio=ratio1)
            self.gpool2 = pygn.TopKPooling(in_channels=feature_dim_out, ratio=ratio2)
        elif Gpool_layer == 'SAG':
            self.gpool1 = pygn.SAGPooling(in_channels=feature_dim_out, ratio=ratio1)
            self.gpool2 = pygn.SAGPooling(in_channels=feature_dim_out, ratio=ratio2)

    def forward(self, x, edge_index, batch):
        """
        :param x: 节点特征 (num_nodes, F)
        :param edge_index: 边索引 (2, num_edges)
        :param batch: 图的 batch 标签 (num_nodes,)
        :return: 池化后的节点特征、边索引和 batch 标签
        """
        x = self.prelu(self.graph_conv1(x=x, edge_index=edge_index))  # 图卷积 + 激活
        x, edge_index, _, batch, _, _ = self.gpool1(x=x, edge_index=edge_index,
                                                     batch=batch)  # 图池化
        x1, _, _, batch2, _, _ = self.gpool2(x=x, edge_index=edge_index, batch=batch)  # 图池化
        x1 = x1.view(torch.max(batch2) + 1, self.ratio2, self.feature_dim_out)
        return x, x1, edge_index, batch




class Mesh_En_rs2_layer(nn.Module):
    def __init__(self, feature_dim_in, feature_dim_out, ratio,
                 GCN_layer='GAT', Gpool_layer='TopK'):
        """
        图神经网络编码层：支持 GCN / GAT 图卷积，支持 TopK / SAG 图池化
        """
        super(Mesh_En_rs2_layer, self).__init__()
        self.prelu = nn.PReLU()

        # 图卷积选择
        if GCN_layer == 'GCN':
            self.graph_conv1 = pygn.GCNConv(in_channels=feature_dim_in,
                                            out_channels=feature_dim_out)
            self.graph_conv2 = pygn.GCNConv(in_channels=feature_dim_out,
                                            out_channels=feature_dim_out)
        elif GCN_layer == 'GAT':
            self.graph_conv1 = pygn.GATv2Conv(in_channels=feature_dim_in,
                                              out_channels=feature_dim_out)
            self.graph_conv2 = pygn.GATv2Conv(in_channels=feature_dim_out,
                                              out_channels=feature_dim_out)

        # 图池化选择
        if Gpool_layer == 'TopK':
            self.gpool = pygn.TopKPooling(in_channels=feature_dim_out, ratio=ratio)

        elif Gpool_layer == 'SAG':
            self.gpool = pygn.SAGPooling(in_channels=feature_dim_out, ratio=ratio)


    def forward(self, x, edge_index, batch):
        """
        :param x: 节点特征 (num_nodes, F)
        :param edge_index: 边索引 (2, num_edges)
        :param batch: 图的 batch 标签 (num_nodes,)
        :return: 池化后的节点特征、边索引和 batch 标签
        """
        x1 = self.prelu(self.graph_conv1(x=x, edge_index=edge_index))  # 图卷积 + 激活
        x2 = self.prelu(self.graph_conv2(x=x1, edge_index=edge_index))  # 图卷积 + 激活
        x, edge_index, _, batch, _, _ = self.gpool(x=x1 + x2, edge_index=edge_index,
                                                   batch=batch)  # 图池化
        return x, edge_index, batch



class Mesh_En_rs3_layer(nn.Module):
    def __init__(self, feature_dim_in, feature_dim_out, ratio,
                 GCN_layer='GAT', Gpool_layer='TopK'):
        """
        图神经网络编码层：支持 GCN / GAT 图卷积，支持 TopK / SAG 图池化
        """
        super(Mesh_En_rs3_layer, self).__init__()
        self.prelu = nn.PReLU()

        # 图卷积选择
        if GCN_layer == 'GCN':
            self.graph_conv1 = pygn.GCNConv(in_channels=feature_dim_in,
                                            out_channels=feature_dim_out)
            self.graph_conv2 = pygn.GCNConv(in_channels=feature_dim_out,
                                            out_channels=feature_dim_out)
            self.graph_conv3 = pygn.GCNConv(in_channels=feature_dim_out,
                                            out_channels=feature_dim_out)
        elif GCN_layer == 'GAT':
            self.graph_conv1 = pygn.GATv2Conv(in_channels=feature_dim_in,
                                              out_channels=feature_dim_out)
            self.graph_conv2 = pygn.GATv2Conv(in_channels=feature_dim_out,
                                              out_channels=feature_dim_out)
            self.graph_conv3 = pygn.GATv2Conv(in_channels=feature_dim_out,
                                              out_channels=feature_dim_out)

        # 图池化选择
        if Gpool_layer == 'TopK':
            self.gpool = pygn.TopKPooling(in_channels=feature_dim_out, ratio=ratio)

        elif Gpool_layer == 'SAG':
            self.gpool = pygn.SAGPooling(in_channels=feature_dim_out, ratio=ratio)


    def forward(self, x, edge_index, batch):
        """
        :param x: 节点特征 (num_nodes, F)
        :param edge_index: 边索引 (2, num_edges)
        :param batch: 图的 batch 标签 (num_nodes,)
        :return: 池化后的节点特征、边索引和 batch 标签
        """
        x1 = self.prelu(self.graph_conv1(x=x, edge_index=edge_index))  # 图卷积 + 激活
        x2 = self.prelu(self.graph_conv2(x=x1, edge_index=edge_index))  # 图卷积 + 激活
        x3 = self.prelu(self.graph_conv2(x=x2, edge_index=edge_index))  # 图卷积 + 激活
        x, edge_index, _, batch, _, _ = self.gpool(x=x1 + x2 + x3, edge_index=edge_index,
                                                   batch=batch)  # 图池化
        return x, edge_index, batch


class POS_ENC(nn.Module):
    def __init__(self, feature_dim_in, feature_dim_out):
        super(POS_ENC, self).__init__()
        self.pos_sin_enc = PositionalEnc()
        self.pos_fc1 = fcBlock(in_planes=feature_dim_in, out_planes=feature_dim_out // 2,
                               bias=False, BN=False, nonlin_ly='prelu')
        self.pos_fc2 = fcBlock(in_planes=feature_dim_out // 2, out_planes=feature_dim_out,
                               bias=False, BN=False, nonlin_ly='prelu')


    def forward(self, pos):
        pos = self.pos_sin_enc(pos)
        pos = self.pos_fc1(pos)
        pos = self.pos_fc2(pos)
        return pos



class ER_Enc(nn.Module):
    def __init__(self, query_seq_length, mesh_enc_dim):
        """
        图神经网络编码层：支持 GCN / GAT 图卷积，支持 TopK / SAG 图池化
        """
        super(ER_Enc, self).__init__()



    def forward(self, x, edge_index, batch):
        """
        :param x: 节点特征 (num_nodes, F)
        :param edge_index: 边索引 (2, num_edges)
        :param batch: 图的 batch 标签 (num_nodes,)
        :return: 池化后的节点特征、边索引和 batch 标签
        """
        x1 = self.prelu(self.graph_conv1(x=x, edge_index=edge_index))  # 图卷积 + 激活
        x2 = self.prelu(self.graph_conv2(x=x1, edge_index=edge_index))  # 图卷积 + 激活
        x3 = self.prelu(self.graph_conv2(x=x2, edge_index=edge_index))  # 图卷积 + 激活
        x, edge_index, _, batch, _, _ = self.gpool(x=x1 + x2 + x3, edge_index=edge_index,
                                                   batch=batch)  # 图池化
        return x, edge_index, batch



def same_padding(kernel_size, stride=1, dilation=1):
    """
    计算保持 H×W 不变所需的 padding（对称填充）。
    仅在 stride 能整除时保证严格等式：out = in。
    """
    if isinstance(kernel_size, int):
        kh = kw = kernel_size
    else:
        kh, kw = kernel_size

    if isinstance(stride, int):
        sh = sw = stride
    else:
        sh, sw = stride

    if isinstance(dilation, int):
        dh = dw = dilation
    else:
        dh, dw = dilation

    # 有效感受野
    eff_kh = (kh - 1) * dh + 1
    eff_kw = (kw - 1) * dw + 1

    # SAME: ph = (eff_kh - sh) / 2 ；要求为整数
    ph = max(eff_kh - sh, 0)
    pw = max(eff_kw - sw, 0)

    # 返回对称 padding（Conv2d 的 padding 参数是每边同样大小）
    return ph // 2, pw // 2



if __name__ == '__main__':
    lst_src_pos = torch.randn(2, 6) * 10
    Pos_Enc = PositionalEnc(16)
    pe = Pos_Enc(lst_src_pos)
    pe_np = (pe.detach().numpy())[0]