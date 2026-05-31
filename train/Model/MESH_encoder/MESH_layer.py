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



class Mesh_En_GCN_layer(nn.Module):
    def __init__(self, feature_dim_in, feature_dim_out, ratio,
                 GCN_layer='GAT', Gpool_layer='TopK', NL='prelu'):
        """
        图神经网络编码层：支持 GCN / GAT 图卷积，支持 TopK / SAG 图池化
        """
        super(Mesh_En_GCN_layer, self).__init__()
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


