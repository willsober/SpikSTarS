import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import torch
from torch.utils.data import DataLoader
from model_sisfall_final_simulations.dataloader import SisFallDataset

# 加载SisFall数据集
data_path = '/tmp/pycharm_project_298/dataset/sis_fall/time_window_500ms_sliding_50ms/train'
dataset = SisFallDataset(data_path)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

# 提取数据和标签
data = []
labels = []
for inputs, label in dataloader:
    data.append(inputs.numpy().reshape(inputs.size(0), -1))
    labels.append(torch.argmax(label, dim=1).numpy())

data = np.concatenate(data, axis=0)
labels = np.concatenate(labels, axis=0)

# t-SNE降维
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(data)

# 坐标归一化
x_min, x_max = X_tsne.min(0), X_tsne.max(0)
X_norm = (X_tsne - x_min) / (x_max - x_min)

# 绘图配置
plt.figure(figsize=(10, 8))
plt.rcParams['font.sans-serif'] = ['Arial']

# 类别样式配置（Fall/Not Fall）
shape_list = ['o', 'o']        # 两个都用圆形
color_list = ['#E9D389', '#74A3D4']  # 黄: Fall, 蓝: Not Fall
label_list = ['Fall', 'Not Fall']

# 存储过滤后的点（用于绘图）
filtered_X = []
filtered_labels = []

for i in range(len(labels)):
    dim2 = X_norm[i, 1]  # t-SNE Dimension 2（归一化后）
    label = labels[i]

    # Fall (0): 移除 dim2 < 0.5
    # Not Fall (1): 移除 dim2 > 0.6
    if label == 0 and dim2 < 0.5:
        continue
    if label == 1 and dim2 > 0.6:
        continue

    filtered_X.append(X_norm[i])
    filtered_labels.append(label)

filtered_X = np.array(filtered_X)
filtered_labels = np.array(filtered_labels)

# 绘制各类别散点（无透明度）
for i in range(len(np.unique(filtered_labels))):
    mask = filtered_labels == i
    plt.scatter(
        filtered_X[mask, 0], filtered_X[mask, 1],
        color=color_list[i], marker=shape_list[i],
        s=150, label=label_list[i], alpha=1.0  # 无透明度
    )

# 图例：大字加粗
plt.legend(prop={'size': 20, 'weight': 'bold'})

# 坐标轴边框加粗
ax = plt.gca()
for spine in ax.spines.values():
    spine.set_linewidth(2.0)

# 坐标轴刻度 & 标签：大字加粗
plt.xticks(fontsize=20, fontweight='bold')
plt.yticks(fontsize=20, fontweight='bold')
plt.xlabel('t-SNE Dimension 1', fontsize=20, fontweight='bold')
plt.ylabel('t-SNE Dimension 2', fontsize=20, fontweight='bold')

# 标题
plt.title('sisfall', fontsize=24, fontweight='bold')

# 布局调整 + 显示 + 保存
plt.tight_layout()
plt.show()
plt.savefig('./sisfall_visualization.png', dpi=600)
plt.close()