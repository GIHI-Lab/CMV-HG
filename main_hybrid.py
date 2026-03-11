"""
Perform training and testing of Hybrid-GCNH on the 10 available splits
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


if __name__ == "__main__":

    args = parse_args()
    cuda = torch.cuda.is_available()

    if args.use_seed:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    n_nodes, n_classes = get_nodes_classes(args.dataset)
    labeled = None
    if args.dataset in ['cora', 'pubmed', 'citeseer']:
        adj, features, labels, idx_train, idx_val, idx_test, labeled = load_data_cit(args.dataset, undirected=True)
    else:
        features, labels, idx_train, idx_val, idx_test = load_data(args.dataset)
        adj = load_graph(args.dataset, n_nodes, features, undirected=True)            

    print("Train percentage: ", len(idx_train) / (len(idx_train) + len(idx_val) + len(idx_test)))
    print("Eval percentage: ", len(idx_val) / (len(idx_train) + len(idx_val) + len(idx_test)))
    print("Test percentage: ", len(idx_test) / (len(idx_train) + len(idx_val) + len(idx_test)))

    tot_splits = 10

    if args.aggfunc not in ["mean", "sum", "maxpool"]:
        print('Valid aggregation functions are "sum", "mean", "maxpool".\nAggregation function "{}" is not available. Using "sum" instead.'.format(args.aggfunc))

    if args.aggfunc == "mean":
        # Mean aggregation requires to normalize the adjacency matrix
        print("Normalizing adj")
        adj = normalize(adj, False)

    if args.aggfunc == "maxpool":
        # Precomputing this allows for a fast execution of maxpooling aggregation
        coo_m =  coo_matrix(adj.numpy())
        row, col = torch.tensor(coo_m.row).long(), torch.tensor(coo_m.col).long()
    else:
        row, col = None, None

    # 创建experiment_results目录
    if not os.path.exists('experiment_results'):
        os.makedirs('experiment_results')

    
    
    split_acc = []
    for split in range(tot_splits):
        
        print("Split: ", split)
        
        idx_train, idx_val, idx_test = load_idx(split, args.dataset, labeled)
        
        model = Hybrid_GCNH(nfeat=features.shape[1],
                           nhid=args.nhid,
                           nclass=n_classes,
                           dropout=args.dropout,
                           nlayers=args.nlayers,
                           maxpool=args.aggfunc == "maxpool",
                        #    k=args.k,
                        #    ppr_threshold=args.ppr_threshold
                        )
        
        if cuda:
            print("Using CUDA")
            model.cuda()
            features = features.cuda()
            adj = adj.cuda()
            labels = labels.cuda()
            idx_train = idx_train.cuda()
            idx_test = idx_test.cuda()
            idx_val = idx_val.cuda()
            if args.aggfunc == "maxpool":
                row, col = row.cuda(), col.cuda()
        
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        batch_size = args.batch_size
        num_batches = len(idx_train) // batch_size + 1
        print("Number of batches: ", num_batches)

        state_dict_early_model = None
        best_val_acc = 0.0
        best_val_loss = 0.0

        t1 = datetime.now()
        if args.verbose:
            epochs = range(args.epochs)
        else:
            epochs = tqdm(range(args.epochs))

        patience_count = 0
        for epoch in epochs:

            if patience_count > args.patience:
                break

            model.train()

            idx = list(range(len(idx_train)))
            np.random.shuffle(idx)
            tot_acc = 0
            tot_loss = 0

            for batch in range(num_batches):
                optimizer.zero_grad()
                cur_idx = idx_train[idx[batch * batch_size: batch * batch_size + batch_size]]
                # For each batch, forward the whole graph but compute loss only on nodes in current batch
                output = model(features, adj, cur_idx=cur_idx, verbose=False, row=row, col=col)
                train_loss = F.nll_loss(output, labels[cur_idx])
                train_acc = accuracy(output, labels[cur_idx])
                train_loss.backward()
                optimizer.step()
                tot_loss += train_loss.detach().cpu().numpy()
                tot_acc += train_acc

            # Validation for each epoch
            model.eval()
            with torch.no_grad():
                output = model(features, adj, cur_idx=idx_val, verbose=False, row=row, col=col)

                val_loss = F.nll_loss(output, labels[idx_val])
                val_acc = accuracy(output, labels[idx_val])

            if args.verbose:
                print(
                    "Epoch {:05d} | Train Loss {:.4f} | Train Acc {:.4f} | Val Loss {:.4f} | Val Acc {:.4f}".format(
                        epoch, train_loss.item(), train_acc, val_loss, val_acc))

            if val_acc >= best_val_acc and (val_acc > best_val_acc or val_loss < best_val_loss):
                best_val_acc = val_acc.cpu()
                best_val_loss = val_loss.detach().cpu()
                state_dict_early_model = deepcopy(model.state_dict())
                patience_count = 0
            else:
                patience_count += 1

        # Perform test
        with torch.no_grad():

            print("Testing")

            model.load_state_dict(state_dict_early_model)
            model.eval()    

            output = model(features, adj, cur_idx=idx_test, verbose=True, row=row, col=col)
            acc_test = accuracy(output, labels[idx_test])

        t2 = datetime.now()
        split_acc.append(acc_test.item())
        print("Test_acc" + ":" + str(acc_test.item()))
        print("Time: ", (t2-t1).total_seconds())
    
    split_acc = np.array(split_acc)
    print("Average acc: ", split_acc.mean())


    # # 遍历不同的k和n_clusters组合
    # k_values = range(2, 11)  # k从2到10
    # n_clusters_values = range(2, 8)  # n_clusters从2到7

    # # 使用追加模式打开结果文件
    # with open('experiment_results/parameter_results.txt', 'a') as f:
    #     # 添加分隔线
    #     f.write("\n" + "="*50 + "\n")
    #     # 记录数据集名称和时间
    #     current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    #     f.write(f"Dataset: {args.dataset}\n")
    #     f.write(f"Time: {current_time}\n\n")
    #     f.write("k\tn_clusters\tAverage Accuracy\n")
        
    #     # 用于记录所有结果
    #     all_results = []
        
    #     # 遍历所有组合
    #     for k in k_values:
    #         for n_clusters in n_clusters_values:
    #             print(f"\nTesting k={k}, n_clusters={n_clusters}")
                
    #             # 重置split_acc
    #             split_acc = []
                
    #             # 对每个split进行训练和测试
    #             for split in range(tot_splits):
    #                 print("Split: ", split)
                    
    #                 # 重新加载数据，确保数据状态重置
    #                 if args.dataset in ['cora', 'pubmed', 'citeseer']:
    #                     adj, features, labels, idx_train, idx_val, idx_test, labeled = load_data_cit(args.dataset, undirected=True)
    #                 else:
    #                     features, labels, idx_train, idx_val, idx_test = load_data(args.dataset)
    #                     adj = load_graph(args.dataset, n_nodes, features, undirected=True)
                    
    #                 # 重新加载split索引
    #                 idx_train, idx_val, idx_test = load_idx(split, args.dataset, labeled)
                    
    #                 # 创建新的模型实例
    #                 model = Hybrid_GCNH(nfeat=features.shape[1],
    #                                   nhid=args.nhid,
    #                                   nclass=n_classes,
    #                                   dropout=args.dropout,
    #                                   nlayers=args.nlayers,
    #                                   maxpool=args.aggfunc == "maxpool",
    #                                   k=k,
    #                                   n_clusters=n_clusters)
                    
    #                 if cuda:
    #                     print("Using CUDA")
    #                     model.cuda()
    #                     features = features.cuda()
    #                     adj = adj.cuda()
    #                     labels = labels.cuda()
    #                     idx_train = idx_train.cuda()
    #                     idx_test = idx_test.cuda()
    #                     idx_val = idx_val.cuda()
    #                     if args.aggfunc == "maxpool":
    #                         row, col = row.cuda(), col.cuda()
                    
    #                 # 创建新的优化器实例
    #                 optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
                    
    #                 batch_size = args.batch_size
    #                 num_batches = len(idx_train) // batch_size + 1
    #                 print("Number of batches: ", num_batches)
                    
    #                 state_dict_early_model = None
    #                 best_val_acc = 0.0
    #                 best_val_loss = 0.0
                    
    #                 t1 = datetime.now()
    #                 if args.verbose:
    #                     epochs = range(args.epochs)
    #                 else:
    #                     epochs = tqdm(range(args.epochs))
                    
    #                 patience_count = 0
    #                 for epoch in epochs:
    #                     if patience_count > args.patience:
    #                         break
                            
    #                     model.train()
    #                     idx = list(range(len(idx_train)))
    #                     np.random.shuffle(idx)
    #                     tot_acc = 0
    #                     tot_loss = 0
                        
    #                     for batch in range(num_batches):
    #                         optimizer.zero_grad()
    #                         cur_idx = idx_train[idx[batch * batch_size: batch * batch_size + batch_size]]
    #                         output = model(features, adj, cur_idx=cur_idx, verbose=False, row=row, col=col)
    #                         train_loss = F.nll_loss(output, labels[cur_idx])
    #                         train_acc = accuracy(output, labels[cur_idx])
    #                         train_loss.backward()
    #                         optimizer.step()
    #                         tot_loss += train_loss.detach().cpu().numpy()
    #                         tot_acc += train_acc
                        
    #                     # 验证
    #                     model.eval()
    #                     with torch.no_grad():
    #                         output = model(features, adj, cur_idx=idx_val, verbose=False, row=row, col=col)
    #                         val_loss = F.nll_loss(output, labels[idx_val])
    #                         val_acc = accuracy(output, labels[idx_val])
                            
    #                         if args.verbose:
    #                             print(
    #                                 "Epoch {:05d} | Train Loss {:.4f} | Train Acc {:.4f} | Val Loss {:.4f} | Val Acc {:.4f}".format(
    #                                     epoch, train_loss.item(), train_acc, val_loss, val_acc))
                            
    #                         if val_acc >= best_val_acc and (val_acc > best_val_acc or val_loss < best_val_loss):
    #                             best_val_acc = val_acc.cpu()
    #                             best_val_loss = val_loss.detach().cpu()
    #                             state_dict_early_model = deepcopy(model.state_dict())
    #                             patience_count = 0
    #                         else:
    #                             patience_count += 1
                    
    #                 # 测试
    #                 with torch.no_grad():
    #                     print("Testing")
    #                     model.load_state_dict(state_dict_early_model)
    #                     model.eval()    
    #                     output = model(features, adj, cur_idx=idx_test, verbose=True, row=row, col=col)
    #                     acc_test = accuracy(output, labels[idx_test])
                    
    #                 t2 = datetime.now()
    #                 split_acc.append(acc_test.item())
    #                 print("Test_acc" + ":" + str(acc_test.item()))
    #                 print("Time: ", (t2-t1).total_seconds())
                    
    #                 # 清理GPU缓存
    #                 if cuda:
    #                     torch.cuda.empty_cache()
                
    #             # 计算平均准确率
    #             avg_acc = np.array(split_acc).mean()
    #             print(f"Average accuracy for k={k}, n_clusters={n_clusters}: {avg_acc:.4f}")
                
    #             # 记录结果
    #             f.write(f"{k}\t{n_clusters}\t{avg_acc:.4f}\n")
    #             # 保存结果用于后续查找最佳参数
    #             all_results.append((k, n_clusters, avg_acc))
        
    #     # 找出最佳结果
    #     best_k, best_n_clusters, best_acc = max(all_results, key=lambda x: x[2])
    #     f.write("\nBest Result:\n")
    #     f.write(f"k={best_k}, n_clusters={best_n_clusters}, Average Accuracy={best_acc:.4f}\n") 