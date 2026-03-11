"""
Define Hybrid-GCNH model with three branches and attention fusion
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import GCNH_layer, GCN_layer, APPNP_layer
from utils import *
import numpy as np
from sklearn.cluster import KMeans

class Hybrid_GCNH(nn.Module):
    def __init__(self, nfeat, nclass, nhid, dropout, nlayers, maxpool, k=8, n_clusters=3, alpha=0.15, ppr_threshold=0.1, use_sparse_graph=True):
        super(Hybrid_GCNH, self).__init__()
        
        self.nfeat = nfeat
        self.nclass = nclass
        self.nhid = nhid
        self.dropout = dropout
        self.nlayers = nlayers
        self.maxpool = maxpool
        self.k = k
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.ppr_threshold = ppr_threshold  # 添加PPR阈值参数
        self.use_sparse_graph = use_sparse_graph  # 是否使用稀疏图表示

        # 原始图分支使用GCNH层
        layer_sizes = [nfeat] + [nhid] * (self.nlayers - 1)
        self.orig_layers = nn.ModuleList([GCNH_layer(layer_sizes[i], nhid, maxpool) for i in range(self.nlayers)])



        # KNN图分支
        # 特征融合层：将原始特征和簇标签特征融合
        self.knn_feat_fusion = nn.Linear(nfeat + n_clusters, nfeat)
        self.init_weights(self.knn_feat_fusion)  # 只初始化融合层

        # KNN图分支使用GCN层
        self.knn_layers = nn.ModuleList([GCN_layer(nfeat, nhid, dropout)])


        # PPR图分支使用APPNP层
        self.ppr_layers_appnp = APPNP_layer(nfeat, nhid, dropout)

        # 视角级注意力机制
        # 参数化映射层
        self.W = nn.Linear(nhid, nhid, bias=True)  # W: [nhid, nhid]
        self.init_weights(self.W)  # 初始化注意力相关参数
        self.b = nn.Parameter(torch.zeros(nhid))    # b: [nhid]
        self.q = nn.Parameter(torch.randn(nhid))    # q: [nhid]

        # MLP for classification
        self.MLPcls = nn.Sequential(
            nn.Linear(self.nhid, nclass),
            nn.LogSoftmax(dim=1)
        )
        self.init_weights(self.MLPcls)

        # 初始化KNN图和PPR图
        self.knn_adj = None
        self.ppr_adj = None
        self.cluster_labels = None
        self.precomputed_clusters = None
        self.fused_feat = None  # 缓存融合后的特征

    def init_weights(self, m):
        """初始化单个模块的权重"""
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                m.bias.data.fill_(0.01)
        elif isinstance(m, nn.Parameter):
            torch.nn.init.xavier_uniform_(m)

    @staticmethod
    def compute_cluster_labels(feat, n_clusters, seed=42):
        """使用K-means对给定特征进行聚类并返回one-hot标签
        
        注意: 始终返回CPU张量,因为后续可能需要numpy转换
        """
        feat_cpu = feat.detach().cpu()
        kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=seed)
        cluster_ids = kmeans.fit_predict(feat_cpu.numpy())
        cluster_ids = torch.from_numpy(cluster_ids).long()
        cluster_one_hot = torch.zeros(feat_cpu.shape[0], n_clusters, dtype=feat_cpu.dtype)
        cluster_one_hot[torch.arange(feat_cpu.shape[0]), cluster_ids] = 1.0
        # 返回CPU张量,避免设备不匹配问题
        return cluster_one_hot

    def perform_clustering(self, feat):
        """默认使用模型指定的簇数量进行聚类"""
        return self.compute_cluster_labels(feat, self.n_clusters)

    def set_cluster_labels(self, cluster_labels):
        """设置预计算的簇标签并清空依赖图结构"""
        if cluster_labels is None:
            self.precomputed_clusters = None
        else:
            self.precomputed_clusters = cluster_labels.detach()
        # 清空所有缓存的图结构和特征
        self.cluster_labels = None
        self.knn_adj = None
        self.ppr_adj = None
        self.fused_feat = None

    def build_knn_graph(self, feat, use_sparse=True):
        """构建KNN图（支持稀疏/稠密）
        Args:
            feat: 特征矩阵 [N, F]
            use_sparse: 是否使用稀疏表示（推荐用于大图）
        Returns:
            adj: 归一化后的邻接矩阵（稀疏或稠密）
        """
        n = feat.shape[0]
        device = feat.device
        
        # 对特征进行L2归一化
        feat_norm = F.normalize(feat, p=2, dim=1)
        
        # 分块计算相似度以节省内存（适用于大图）
        batch_size = min(1000, n)
        edge_index = []
        edge_weight = []
        
        for i in range(0, n, batch_size):
            end_i = min(i + batch_size, n)
            # 计算当前批次与所有节点的相似度
            sim_batch = torch.mm(feat_norm[i:end_i], feat_norm.t())
            
            # 获取top-k个最近邻
            topk_sim, topk_indices = torch.topk(sim_batch, k=self.k, largest=True, dim=1)
            
            # 收集边
            for local_idx in range(end_i - i):
                global_idx = i + local_idx
                for k_idx in range(self.k):
                    neighbor = topk_indices[local_idx, k_idx].item()
                    sim_value = topk_sim[local_idx, k_idx].item()
                    
                    # 添加双向边（对称化）
                    edge_index.append([global_idx, neighbor])
                    edge_weight.append(sim_value)
                    if global_idx != neighbor:
                        edge_index.append([neighbor, global_idx])
                        edge_weight.append(sim_value)
        
        if use_sparse:
            # 构建稀疏COO格式
            edge_index = torch.tensor(edge_index, device=device).t()
            edge_weight = torch.tensor(edge_weight, device=device)
            
            # 去重并平均权重（对称化可能产生重复边）
            adj_sparse = torch.sparse_coo_tensor(
                edge_index, edge_weight, (n, n), device=device
            ).coalesce()
            
            # 归一化（度归一化）
            adj_sparse = normalize(adj_sparse, is_sparse=True)
            return adj_sparse
        else:
            # 构建稠密矩阵（兼容原有逻辑）
            adj = torch.zeros(n, n, device=device)
            edge_index = torch.tensor(edge_index, device=device)
            adj[edge_index[:, 0], edge_index[:, 1]] = 1.0
            adj = normalize(adj, is_sparse=False)
            return adj

    def compute_cluster_aware_ppr(self, adj, cluster_labels, use_sparse=True):
        """计算簇感知PPR（支持稀疏优化）
        Args:
            adj: 原始邻接矩阵
            cluster_labels: 簇标签 [N, C]
            use_sparse: 是否使用稀疏矩阵优化
        Returns:
            S: PPR扩散矩阵
        """
        n = adj.shape[0]
        device = adj.device
        
        # 批量计算节点簇分配
        node_clusters = torch.argmax(cluster_labels, dim=1)
        
        # 批量构建重启矩阵（保持稠密，因为通常较小）
        cluster_indices = node_clusters.unsqueeze(1) == node_clusters.unsqueeze(0)  # [N, N]
        cluster_sizes = cluster_indices.sum(dim=1, keepdim=True)
        R = cluster_indices.float() / cluster_sizes
        
        # 归一化邻接矩阵
        T = normalize(adj, is_sparse=adj.is_sparse if hasattr(adj, 'is_sparse') else False)
        
        # 使用矩阵乘法计算PPR
        S = R.clone()
        X = R.clone()
        for _ in range(10):
            if T.is_sparse:
                X = (1 - self.alpha) * torch.sparse.mm(T, X)
            else:
                X = (1 - self.alpha) * torch.mm(T, X)
            S = S + X
        S = self.alpha * S
        
        # 应用阈值进行稀疏化
        mask = S < self.ppr_threshold
        S[mask] = 0.0
        
        if use_sparse and (S != 0).sum() / (n * n) < 0.1:  # 稀疏度>90%时转稀疏
            # 转换为稀疏格式
            S_sparse = S.to_sparse()
            # 重新进行行归一化
            indices = S_sparse.indices()
            values = S_sparse.values()
            
            row_sum = torch.sparse.sum(S_sparse, dim=1).to_dense()
            row_sum = torch.where(row_sum == 0, torch.ones_like(row_sum), row_sum)
            
            # 归一化边权重
            normalized_values = values / row_sum[indices[0]]
            S_normalized = torch.sparse_coo_tensor(
                indices, normalized_values, S.shape, device=device
            ).coalesce()
            return S_normalized
        else:
            # 保持稠密格式
            row_sum = S.sum(dim=1, keepdim=True)
            row_sum = torch.where(row_sum == 0, torch.ones_like(row_sum), row_sum)
            S = S / row_sum
            return S
    
    def init_graphs(self, feat, adj, use_sparse=True):
        """初始化KNN图和PPR图
        Args:
            feat: 特征矩阵
            adj: 原始邻接矩阵
            use_sparse: 是否使用稀疏图结构（推荐）
        """
        # 执行聚类或使用预先计算的簇标签
        if self.precomputed_clusters is not None:
            self.cluster_labels = self.precomputed_clusters.to(feat.device)
        else:
            self.cluster_labels = self.perform_clustering(feat).to(feat.device)
        
        # 特征融合：将原始特征和簇标签特征融合（只在init时做一次）
        with torch.no_grad():  # 在no_grad上下文中进行初始化，避免计算图问题
            fused_feat = torch.cat([feat, self.cluster_labels], dim=1)
            fused_feat = self.knn_feat_fusion(fused_feat)
            
            # 保存融合后的特征，避免在forward中重复计算
            self.fused_feat = fused_feat
            
            # 构建KNN图（使用稀疏格式）
            self.knn_adj = self.build_knn_graph(fused_feat, use_sparse=use_sparse)
            
            # 计算簇感知PPR扩散矩阵（使用稀疏格式）
            self.ppr_adj = self.compute_cluster_aware_ppr(adj, self.cluster_labels, use_sparse=use_sparse)

    def forward(self, feat, adj, cur_idx=None, verbose=False, row=None, col=None):
        """
        feat: feature matrix 
        adj: original adjacency matrix
        cur_idx: index of nodes in current batch
        row, col: used for maxpool aggregation
        """
        if cur_idx == None:
            cur_idx = range(feat.shape[0])

        # 如果KNN图和PPR图还没有初始化，则初始化它们
        if self.knn_adj is None or self.ppr_adj is None or self.cluster_labels is None:
            self.init_graphs(feat, adj, use_sparse=self.use_sparse_graph)

        # 原始图分支 (GCNH)
        hp_orig = feat
        for i in range(self.nlayers):
            hp_orig, beta = self.orig_layers[i](hp_orig, adj, cur_idx=cur_idx, row=row, col=col)
            if verbose:
                print("Original Layer: ", i, " beta: ", beta.item())
            hp_orig = F.dropout(hp_orig, self.dropout, training=self.training)



        # KNN图分支 
        # 使用预先融合好的特征（在init_graphs中已完成融合）
        hp_knn = self.knn_layers[0](self.fused_feat, self.knn_adj)

        # 使用APPNP层处理ppr
        hp_ppr = self.ppr_layers_appnp(feat, self.ppr_adj)

        # 视角级注意力机制
        # 1. 对每个视角的嵌入进行参数化映射
        h_orig = torch.tanh(self.W(hp_orig) + self.b)  # [N, nhid]
        h_knn = torch.tanh(self.W(hp_knn) + self.b)    # [N, nhid]
        h_ppr = torch.tanh(self.W(hp_ppr) + self.b)    # [N, nhid]
        
        # 2. 计算每个视角的重要性得分
        s_orig = torch.sum(self.q * h_orig, dim=1)  # [N]
        s_knn = torch.sum(self.q * h_knn, dim=1)    # [N]
        s_ppr = torch.sum(self.q * h_ppr, dim=1)    # [N]
        
        # 3. 对所有视角的得分进行Softmax归一化
        s = torch.stack([s_orig, s_knn, s_ppr], dim=1)  # [N, 3]
        beta = F.softmax(s, dim=1)  # [N, 3]
        
        # 4. 加权融合
        hp_fused = beta[:, 0].unsqueeze(1) * hp_orig + \
                   beta[:, 1].unsqueeze(1) * hp_knn + \
                   beta[:, 2].unsqueeze(1) * hp_ppr

        return self.MLPcls(hp_fused[cur_idx])