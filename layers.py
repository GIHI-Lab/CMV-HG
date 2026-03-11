"""
GCNH Layer
"""

import torch
import torch.nn as nn
from torch.nn.modules.module import Module
from torch_scatter import scatter
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import math


class GCNH_layer(Module):
    def __init__(self, nfeat, nhid, maxpool):
        super(GCNH_layer, self).__init__()
        
        self.nhid = nhid
        self.maxpool = maxpool
        
        # Two MLPs, one to encode center-node embedding,
        # the other for the neighborhood embedding
        self.MLPfeat = nn.Sequential(
            nn.Linear(nfeat, self.nhid),
            nn.LeakyReLU()
        )
        self.init_weights(self.MLPfeat)
        
        self.MLPmsg = nn.Sequential(
            nn.Linear(nfeat, self.nhid),
            nn.LeakyReLU()
        )
        self.init_weights(self.MLPmsg)
        
        # Parameter beta
        self.beta = nn.Parameter(0.0 * torch.ones(size=(1, 1)), requires_grad=True) 
    
    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform(m.weight)
            m.bias.data.fill_(0.01)
            
    def forward(self, feat, adj, cur_idx=None,row=None,col=None):
        """
        feat: feature matrix 
        adj: adjacency matrix
        cur_idx: index of nodes in current batch
        row, col: used for maxpool aggregation
        """
        if cur_idx == None:
            cur_idx = range(feat.shape[0])
        
        # Transform center-node and neighborhood messages
        h = self.MLPfeat(feat)
        z = self.MLPmsg(feat)
        
        # Aggregate messages
        beta = torch.sigmoid(self.beta)
        
        if not self.maxpool: # sum or mean
            hp = beta * z + (1-beta) * torch.matmul(adj, h)
        else:
            hh = torch.zeros(adj.shape[0], self.nhid)
            if next(self.parameters()).is_cuda:
                hh = hh.cuda()
            _ = scatter(h[row], col, dim=0, out=hh, reduce="max")
            hp = beta * z + (1 - beta) * hh
        
        return hp, beta

class GraphConv(Module):
    """基础图卷积层"""
    def __init__(self, in_features, out_features, bias=False):
        super(GraphConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

class GCN_layer(Module):
    """由两个图卷积层组成的GCN层"""
    def __init__(self, nfeat, nhid, dropout=0.0):
        super(GCN_layer, self).__init__()
        
        self.nhid = nhid
        self.dropout = dropout
        
        # 第一层卷积：输入维度nfeat，输出维度nhid
        self.gc1 = GraphConv(nfeat, nhid)
        # 第二层卷积：输入维度nhid，输出维度nhid
        self.gc2 = GraphConv(nhid, nhid)
        
        self.init_weights()
    
    def init_weights(self):
        self.gc1.reset_parameters()
        self.gc2.reset_parameters()
            
    def forward(self, feat, adj):
        """
        feat: feature matrix 
        adj: adjacency matrix (已经归一化)
        """
        # 第一层GCN
        h = self.gc1(feat, adj)
        h = F.relu(h)
        h = F.dropout(h, self.dropout, training=self.training)
        
        # 第二层GCN
        h = self.gc2(h, adj)
        h = F.relu(h)
        h = F.dropout(h, self.dropout, training=self.training)
        
        return h

class APPNP_layer(Module):
    """APPNP层实现"""
    def __init__(self, nfeat, nhid, dropout=0.0, alpha=0.1, K=10):
        super(APPNP_layer, self).__init__()
        
        self.nhid = nhid
        self.dropout = dropout
        self.alpha = alpha  # 重启概率
        self.K = K         # 传播步数
        
        # 特征变换层
        self.lin1 = nn.Linear(nfeat, nhid)
        self.lin2 = nn.Linear(nhid, nhid)
        
        self.init_weights()
    
    def init_weights(self):
        """初始化权重"""
        torch.nn.init.xavier_uniform_(self.lin1.weight)
        self.lin1.bias.data.fill_(0.01)
        torch.nn.init.xavier_uniform_(self.lin2.weight)
        self.lin2.bias.data.fill_(0.01)
            
    def forward(self, feat, adj):
        """
        feat: 输入特征矩阵
        adj: 归一化的邻接矩阵
        """
        # 神经网络特征变换
        h = F.dropout(feat, p=self.dropout, training=self.training)
        h = F.relu(self.lin1(h))
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.lin2(h)
        
        # 初始化传播
        h0 = h
        
        # 迭代传播K步
        for k in range(self.K):
            # 消息传递
            h = (1 - self.alpha) * torch.matmul(adj, h) + self.alpha * h0
        
        return h
