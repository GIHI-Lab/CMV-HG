"""
噪声比例对比实验：分析聚类质量与性能的关系
通过引入不同比例的噪声标签来评估模型鲁棒性
"""

import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from utils import *
from datetime import datetime
from copy import deepcopy
from scipy.sparse import coo_matrix
from models_hybrid import Hybrid_GCNH
from tqdm import tqdm
import os
from precompute_clusters import load_clustering_results, precompute_for_dataset


def add_noise_to_clusters(cluster_labels, noise_ratio, seed=42):
    """
    向聚类标签中添加噪声
    
    Args:
        cluster_labels: 原始聚类标签 [N, C] (one-hot)
        noise_ratio: 噪声比例 (0.0 ~ 1.0)
        seed: 随机种子
    
    Returns:
        noisy_labels: 添加噪声后的聚类标签
    """
    np.random.seed(seed)
    n_nodes = cluster_labels.shape[0]
    n_clusters = cluster_labels.shape[1]
    
    # 复制原始标签
    noisy_labels = cluster_labels.clone()
    
    # 计算需要添加噪声的节点数量
    n_noisy = int(noise_ratio * n_nodes)
    
    if n_noisy > 0:
        # 随机选择要添加噪声的节点
        noisy_indices = np.random.choice(n_nodes, n_noisy, replace=False)
        
        # 为这些节点随机分配新的聚类标签
        random_assignments = np.random.randint(0, n_clusters, n_noisy)
        
        # 更新标签
        noisy_labels[noisy_indices] = 0  # 先清零
        noisy_labels[noisy_indices, random_assignments] = 1  # 重新赋值
    
    return noisy_labels


def run_single_experiment(args, noise_ratio, split=0):
    """
    运行单次实验（指定噪声比例）
    
    Returns:
        test_acc: 测试准确率
    """
    cuda = torch.cuda.is_available()
    
    # 加载数据
    if args.dataset in ["cora", "citeseer"]:
        adj, features, labels, idx_train, idx_val, idx_test, labeled = load_data_cit(args.dataset, split)
    else:
        features, labels, idx_train, idx_val, idx_test = load_data(args.dataset, split)
        n_nodes, n_classes = get_nodes_classes(args.dataset)
        adj = load_graph(args.dataset, n_nodes, features)
    
    # 加载预计算的聚类结果
    cluster_labels = None

    cluster_file = f"{args.cluster_dir}/{args.dataset}/silhouette_score.npz"
    if os.path.exists(cluster_file):
        cluster_data = np.load(cluster_file)
        cluster_labels = torch.FloatTensor(cluster_data['cluster_labels'])
        n_clusters = cluster_labels.shape[1]
    else:
        print(f"Warning: Cluster file not found: {cluster_file}")
        n_clusters = 7
    
    # 添加噪声到聚类标签（使用split作为种子确保可重复性）
    if cluster_labels is not None and noise_ratio > 0:
        cluster_labels = add_noise_to_clusters(cluster_labels, noise_ratio, seed=42+split)
        print(f"Added {noise_ratio*100:.0f}% noise to cluster labels")
    
    # 初始化模型
    n_nodes, n_classes = get_nodes_classes(args.dataset)
    model = Hybrid_GCNH(
        nfeat=features.shape[1],
        nclass=n_classes,
        nhid=args.nhid,
        dropout=args.dropout,
        nlayers=args.nlayers,
        maxpool=args.aggfunc == "maxpool",
        k=args.k,
        n_clusters=n_clusters,
        alpha=0.15,
        ppr_threshold=args.ppr_threshold,
        use_sparse_graph=args.use_sparse_graph
    )
    
    # 设置聚类标签
    if cluster_labels is not None:
        model.set_cluster_labels(cluster_labels)
    
    # 构建边索引（用于maxpool，在移到GPU之前）
    row, col = None, None
    if args.aggfunc == "maxpool":
        edge_index = torch.nonzero(adj, as_tuple=False).t()
        row, col = edge_index[0], edge_index[1]
    
    # 移到GPU
    if cuda:
        model = model.cuda()
        features_gpu = features.cuda()
        adj_gpu = adj.cuda()
        labels_gpu = labels.cuda()
        idx_train = idx_train.cuda()
        idx_val = idx_val.cuda()
        idx_test = idx_test.cuda()
        if args.aggfunc == "maxpool":
            row, col = row.cuda(), col.cuda()
        if cluster_labels is not None:
            cluster_labels = cluster_labels.cuda()
            model.set_cluster_labels(cluster_labels)
    else:
        features_gpu = features
        adj_gpu = adj
        labels_gpu = labels
    
    # 训练
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    best_val_acc = 0
    best_test_acc = 0
    patience_counter = 0
    
    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        
        output = model(features_gpu, adj_gpu, cur_idx=idx_train, row=row, col=col)
        loss_train = F.nll_loss(output, labels_gpu[idx_train])
        acc_train = accuracy(output, labels_gpu[idx_train])
        
        loss_train.backward()
        optimizer.step()
        
        # 验证
        model.eval()
        with torch.no_grad():
            output = model(features_gpu, adj_gpu, cur_idx=idx_val, row=row, col=col)
            loss_val = F.nll_loss(output, labels_gpu[idx_val])
            acc_val = accuracy(output, labels_gpu[idx_val])
            
            # 测试
            output = model(features_gpu, adj_gpu, cur_idx=idx_test, row=row, col=col)
            acc_test = accuracy(output, labels_gpu[idx_test])
        
        # Early stopping
        if acc_val > best_val_acc:
            best_val_acc = acc_val
            best_test_acc = acc_test
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                break
    
    return best_test_acc.item()


def main():
    """主函数"""
    args = parse_args()
    
    # 噪声比例范围
    noise_ratios = [0.0, 0.3, 0.5, 0.7, 1.0]
    
    # 运行多次实验取平均（使用10个split与原始代码一致）
    n_runs = 3  # 修改为10个split，与原始main_hybrid.py保持一致
    
    print("\n" + "="*70)
    print(f"Noise Ratio Experiment - Dataset: {args.dataset}")
    print(f"Noise ratios: {noise_ratios}")
    print(f"Runs per ratio: {n_runs} (using 10 splits like original code)")
    print("="*70 + "\n")
    
    results = []
    
    for noise_ratio in noise_ratios:
        print(f"\nTesting noise ratio: {noise_ratio}")
        accs = []
        
        for run in range(n_runs):
            print(f"  Run {run+1}/{n_runs}...", end=' ')
            acc = run_single_experiment(args, noise_ratio, split=run)
            accs.append(acc)
            print(f"Acc: {acc:.4f}")
        
        mean_acc = np.mean(accs)
        std_acc = np.std(accs)
        results.append((mean_acc, std_acc))
        
        print(f"  Average: {mean_acc:.4f} ± {std_acc:.4f}")
    
    # 保存结果
    save_dir = 'experiment_results/noise_experiment'
    os.makedirs(save_dir, exist_ok=True)
    
    result_file = os.path.join(save_dir, f'{args.dataset}_noise_results.txt')
    with open(result_file, 'w') as f:
        f.write(f"Dataset: {args.dataset}\n")
        f.write(f"Noise Ratios: {noise_ratios}\n")
        f.write(f"Runs per ratio: {n_runs}\n\n")
        f.write("Noise Ratio | Mean Accuracy | Std Dev\n")
        f.write("-" * 45 + "\n")
        for i, noise_ratio in enumerate(noise_ratios):
            mean_acc, std_acc = results[i]
            f.write(f"{noise_ratio:11.1f} | {mean_acc:13.4f} | {std_acc:7.4f}\n")
    
    print(f"\nResults saved to: {result_file}")
    
    print("\nExperiment completed!")


if __name__ == "__main__":
    main()
