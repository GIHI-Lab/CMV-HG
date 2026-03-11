# CMV-HG: 聚类增强多视图异质图表示学习

本仓库包含 CMV-HG 的 PyTorch 实现。CMV-HG 面向异质图节点分类任务，将聚类分析引入图构建阶段，用聚类所提供的潜在语义分组信息增强图结构，再通过多视图编码与节点级注意力融合，联合利用原始拓扑、局部相似关系和高阶扩散依赖。

## 方法概述

CMV-HG 的核心思想是先利用节点特征进行聚类，将聚类结果视为一种语义先验，再据此构建三个互补视图：

- 原始拓扑视图：保留图中直接连接关系，用于维持基础结构信息。
- 局部聚焦视图：基于聚类增强的 KNN 图，强调特征相似节点之间的局部关联。
- 全局扩散视图：基于聚类增强的 PPR 扩散矩阵，建模更高阶的结构依赖。

针对三种视图的结构特性，模型分别使用不同编码器：

- 原始拓扑视图使用 GCNH。
- 局部聚焦视图使用 GCN。
- 全局扩散视图使用 APPNP。

最终，CMV-HG 通过节点级跨视图注意力机制，对三个分支的表示进行自适应加权融合，并输出节点分类结果。

## 模型结构

当前实现位于 models_hybrid.py，对应的主要流程如下：

1. 对输入特征做 K-Means 聚类，得到 one-hot 聚类标签。
2. 将原始特征与聚类标签拼接，并线性映射后构建聚类增强 KNN 图。
3. 基于原始图与聚类标签构建聚类感知 PPR 扩散矩阵。
4. 分别通过 GCNH、GCN、APPNP 三个分支提取节点表示。
5. 用注意力机制学习每个节点在三个视图下的融合权重。
6. 将融合后的表示送入分类头，输出对数概率。

## 仓库结构

```text
CMV-HG/
├── data/                      # 异质图与引文网络数据集
├── experiments/               # 复现实验脚本
├── figures/                   # 图像与中间结果
├── dataset.py                 # 合成数据读取封装
├── layers.py                  # GCNH / GCN / APPNP 层定义
├── main_hybrid.py             # 主训练入口，10 个 split 上训练与测试
├── main_noise_experiment.py   # 聚类噪声鲁棒性实验
├── main_syn.py                # 合成数据实验
├── models_hybrid.py           # CMV-HG 模型定义
├── requirements.txt           # Python 依赖
└── utils.py                   # 参数、数据加载、归一化与评估工具
```

## 环境要求

- Python 3.9 或兼容版本
- PyTorch 2.0.0
- 其余依赖见 requirements.txt

推荐使用虚拟环境后安装依赖：

```bash
pip install -r requirements.txt
```

requirements.txt 当前包含的主要依赖：

- torch==2.0.0
- numpy==1.22.4
- scipy==1.8.1
- scikit_learn==1.1.1
- networkx==2.8.5
- matplotlib==3.5.2
- tqdm==4.64.0

## 支持的数据集

当前仓库已经包含以下数据：

- WebKB: cornell, texas, wisconsin
- WikipediaNetwork / Actor: chameleon, squirrel, film
- Citation Networks: cora, citeseer
- Synthetic: syn-cora

其中：

- 异质图和引文网络默认使用固定的 10 个数据划分。
- 合成数据位于 syn-cora 目录，用于分析不同同质率下的模型行为。

## 快速开始

### 1. 运行主实验

主入口为 main_hybrid.py，会在 10 个 split 上训练并输出平均准确率：

```bash
python main_hybrid.py --dataset cornell
```

也可以直接使用 experiments/main_table.sh 中给出的论文风格配置：

```bash
python main_hybrid.py --dataset cornell --epochs 300 --nlayers 1 --verbose False --dropout 0.25 --nhid 16 --batch_size 50
python main_hybrid.py --dataset texas --epochs 300 --nlayers 1 --verbose False --dropout 0.25 --nhid 32 --batch_size 200
python main_hybrid.py --dataset wisconsin --epochs 300 --nlayers 2 --verbose False --dropout 0.3 --nhid 32 --batch_size 50
python main_hybrid.py --dataset film --epochs 150 --nlayers 2 --verbose False --dropout 0.6 --nhid 32 --batch_size 500
python main_hybrid.py --dataset chameleon --epochs 1000 --nlayers 1 --verbose False --dropout 0.0 --nhid 32 --batch_size 300
python main_hybrid.py --dataset squirrel --epochs 1500 --nlayers 1 --verbose False --dropout 0.0 --nhid 32 --batch_size 1400
python main_hybrid.py --dataset cora --epochs 300 --nlayers 2 --verbose False --dropout 0.75 --nhid 64 --batch_size 150
python main_hybrid.py --dataset citeseer --epochs 300 --nlayers 1 --verbose False --dropout 0.25 --nhid 16 --batch_size 300
```

### 2. 运行合成数据实验

```bash
python main_syn.py --epochs 300 --nlayers 2 --verbose False --dropout 0.75 --nhid 64 --batch_size 150
```

### 3. 运行聚类噪声鲁棒性实验

```bash
python main_noise_experiment.py --dataset cornell
```

## 命令行参数

当前 parse_args() 在 utils.py 中暴露的主要参数如下：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| --dataset | cornell | 数据集名称 |
| --epochs | 100 | 训练轮数 |
| --patience | 1000 | Early stopping 容忍轮数 |
| --batch_size | 100000 | 批大小 |
| --nhid | 16 | 隐层维度 |
| --dropout | 0.0 | Dropout 比例 |
| --nlayers | 1 | GCNH 分支层数 |
| --lr | 5e-3 | 学习率 |
| --weight_decay | 5e-3 | 权重衰减 |
| --aggfunc | sum | 邻居聚合方式，可选 sum / mean / maxpool |
| --seed | 112 | 随机种子 |
| --use_seed | True | 是否启用固定随机种子 |
| --verbose | False | 是否打印每轮训练信息 |

说明：

- 模型内部支持 k、n_clusters、alpha、ppr_threshold、use_sparse_graph 等超参数。
- 但当前 utils.py 中对应的命令行参数并未全部开放，如果需要从命令行调参，需要先补充参数解析逻辑。

## 输出结果

### 主实验

main_hybrid.py 会输出：

- 每个 split 的测试准确率
- 训练耗时
- 10 个 split 的平均准确率

默认会创建 experiment_results 目录用于保存后续实验结果。

### 噪声实验

main_noise_experiment.py 会在 experiment_results/noise_experiment 下保存：

- 各噪声比例对应的平均准确率
- 标准差统计结果

### 合成数据实验

main_syn.py 会将不同同质率下的结果汇总写入结果文件。

## 代码说明

### main_hybrid.py

标准训练与测试入口。对每个 split：

1. 读取图结构、特征和标签。
2. 构建 Hybrid_GCNH 模型。
3. 在训练集上优化，在验证集上选取最优模型。
4. 在测试集上报告性能。

### models_hybrid.py

核心模型实现，包含以下关键模块：

- 原始图分支：GCNH_layer
- KNN 视图分支：GCN_layer
- PPR 视图分支：APPNP_layer
- 聚类标签生成：KMeans
- 聚类增强 KNN 图构建
- 聚类感知 PPR 计算
- 跨视图注意力融合

### utils.py

提供以下公共功能：

- 命令行参数解析
- 不同数据集的加载逻辑
- 邻接矩阵归一化
- 固定 split 读取
- 准确率计算

## 引用

如果本项目对你的研究有帮助，可以在你的论文或报告中引用对应方法与实现，并在使用时注明本仓库为 CMV-HG 的实验代码实现。
