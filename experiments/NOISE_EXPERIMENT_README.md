# 噪声比例对比实验

## 实验目的
通过引入不同比例的噪声标签到聚类结果中，分析聚类质量对模型性能的影响，绘制"噪声比例曲线"。

## 实验原理
- 在聚类标签中随机选择一定比例的节点，重新随机分配簇标签
- 噪声比例 α ∈ {0.0, 0.2, 0.4, 0.6, 0.8, 1.0}
- α = 0.0: 使用原始聚类结果（无噪声）
- α = 1.0: 完全随机的聚类标签（最大噪声）
- 观察随着噪声增加，模型性能的下降趋势

## 文件说明

### 核心实验文件
1. **main_noise_experiment.py** - 主实验脚本
   - 功能：对单个数据集运行噪声比例实验
   - 特点：完全独立，不修改原始代码
   - 输出：每个噪声比例的测试准确率

2. **plot_noise_curves.py** - 结果汇总脚本
   - 功能：将所有数据集的噪声曲线绘制到一张图上
   - 输出：论文风格的对比图

### 运行脚本
3. **experiments/noise_experiment.sh** - Linux/Mac批处理脚本
4. **experiments/noise_experiment.bat** - Windows批处理脚本

## 使用方法

### 方法1：单个数据集实验
```bash
python main_noise_experiment.py --dataset cornell
```

### 方法2：批量运行所有数据集（推荐）

**Windows:**
```cmd
cd experiments
noise_experiment.bat
```

**Linux/Mac:**
```bash
cd experiments
chmod +x noise_experiment.sh
./noise_experiment.sh
```

### 方法3：自定义参数运行
```bash
python main_noise_experiment.py \
    --dataset cora \
    --epochs 100 \
    --patience 50 \
    --nhid 16 \
    --dropout 0.0 \
    --lr 5e-3 \
    --weight_decay 5e-3 \
    --cluster_metric silhouette_score
```

## 结果输出

### 文件结构
```
experiment_results/noise_experiment/
├── cornell_noise_results.txt      # 数值结果
├── cornell_noise_curve.png         # 单数据集曲线图
├── cornell_noise_curve.pdf         # PDF版本
├── texas_noise_results.txt
├── texas_noise_curve.png
├── texas_noise_curve.pdf
├── ...
├── all_datasets_noise_curves.png   # 汇总图（所有数据集）
└── all_datasets_noise_curves.pdf   # PDF版本
```

### 生成汇总图
运行完所有数据集后：
```bash
python plot_noise_curves.py
```

### 结果文件示例
```
Dataset: cornell
Noise Ratios: [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
Runs per ratio: 3

Noise Ratio | Mean Accuracy | Std Dev
---------------------------------------------
        0.0 |        0.8243 |  0.0123
        0.2 |        0.8015 |  0.0156
        0.4 |        0.7621 |  0.0187
        0.6 |        0.7102 |  0.0201
        0.8 |        0.6543 |  0.0234
        1.0 |        0.5876 |  0.0267
```

## 实验参数说明

### 默认噪声比例
```python
noise_ratios = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
```

### 每个比例的运行次数
```python
n_runs = 3  # 取平均值和标准差
```

### 修改参数
在 `main_noise_experiment.py` 中修改：
```python
def main():
    # 修改噪声比例
    noise_ratios = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    
    # 修改运行次数
    n_runs = 5  # 增加运行次数以获得更稳定的结果
```

## 与原始代码的关系

### 完全解耦
- **不修改原始文件**: main_hybrid.py、models_hybrid.py等原始文件保持不变
- **独立运行**: 噪声实验脚本可以独立运行
- **随时切换**: 可以随时切换回原始代码运行标准实验

### 复现原始结果
运行标准实验（无噪声）：
```bash
python main_hybrid.py --dataset cornell
```

### 复现对比实验
运行噪声实验：
```bash
python main_noise_experiment.py --dataset cornell
```

## 预期结果

### 理论预期
1. **α = 0.0**: 最高准确率（使用高质量聚类）
2. **α 增加**: 准确率逐渐下降
3. **α = 1.0**: 最低准确率（完全随机聚类）

### 曲线特征
- **平滑下降**: 说明模型对聚类质量敏感度适中
- **急剧下降**: 说明模型高度依赖聚类质量
- **缓慢下降**: 说明模型对聚类噪声具有一定鲁棒性

## 注意事项

1. **确保预计算聚类存在**
   ```bash
   python precompute_clusters.py --dataset cornell
   ```

2. **计算资源**: 运行所有数据集需要较长时间
   - 每个数据集 6 个噪声比例 × 3 次运行 = 18 次训练
   - 8 个数据集 = 144 次训练

3. **GPU推荐**: 使用GPU可显著加速实验

4. **结果保存**: 所有结果自动保存，可以分批运行

## 论文图表使用

生成的PDF文件可直接用于论文：
- 高分辨率 (300 DPI)
- Times New Roman字体
- 标准学术风格
- 误差带显示标准差

## 故障排除

### 问题1: 找不到聚类文件
```bash
# 解决方法：先运行聚类预计算
python precompute_clusters.py --dataset cornell
```

### 问题2: 内存不足
```python
# 在main_noise_experiment.py中减少运行次数
n_runs = 1  # 从3改为1
```

### 问题3: 想修改噪声范围
```python
# 在main_noise_experiment.py的main()函数中修改
noise_ratios = [0.0, 0.3, 0.5, 0.7, 1.0]  # 自定义范围
```

## 扩展实验

### 1. 更细粒度的噪声比例
```python
noise_ratios = np.linspace(0, 1, 11)  # [0.0, 0.1, 0.2, ..., 1.0]
```

### 2. 不同聚类指标
```bash
python main_noise_experiment.py --dataset cornell --cluster_metric davies_bouldin_score
```

### 3. 不同k值
```bash
python main_noise_experiment.py --dataset cornell --k 5
```

## 致谢
本实验设计参考了聚类质量与模型性能关系的研究范式，通过噪声注入方法定量分析了聚类在图神经网络中的作用。
