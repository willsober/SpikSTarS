import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import torch
from torch.utils.data import DataLoader
from dataloader import SmartFallDataset  # 确保导入正确

# 加载SmartFall数据集
dataset = SmartFallDataset('/tmp/pycharm_project_298/dataset/smart_fall/time_window_1s_sliding_100ms/train')
train_loader = DataLoader(dataset, batch_size=1, shuffle=True)

# 提取数据和标签
data = []
labels = []
for inputs, label in train_loader:
    data.append(inputs.numpy().reshape(inputs.size(0), -1))
    labels.append(torch.argmax(label, dim=1).numpy())

data = np.concatenate(data, axis=0)
labels = np.concatenate(labels, axis=0)

# t-SNE降维（2维）
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(data)

# 坐标归一化
x_min, x_max = X_tsne.min(0), X_tsne.max(0)
X_norm = (X_tsne - x_min) / (x_max - x_min)

# 绘图基础配置
plt.figure(figsize=(10, 8))
plt.rcParams['font.sans-serif'] = ['Arial']

# ---------------------- 样式配置 ----------------------
shape_list = ['o'] * 6  # 所有类别都用圆形
color_list = [
    '#E9D389',  # 黄色调 (Fall)
    '#74A3D4',  # 蓝色调 (Not Fall)
    '#A7D398',  # 绿色调 (Walk)
    '#9A8CD1',  # 紫色调 (Sit)
    '#8CD1CA',  # 青色调 (Stand)
    '#DD847E'   # 红色调 (Lay)
]
label_list = ['Fall', 'Not Fall', 'Walk', 'Sit', 'Stand', 'Lay']

# 绘制各类别散点（圆形 + 无透明度）
unique_labels = np.unique(labels)
for i, label in enumerate(unique_labels):
    plt.scatter(
        X_norm[labels == label, 0], X_norm[labels == label, 1],
        color=color_list[i % len(color_list)],
        marker='o',                    # 强制使用圆形
        s=300,                         # 散点更大（原150 → 300）
        label=label_list[i],
        alpha=1.0                      # 无透明度
    )

# ---------------------- 统一字号+加粗设置 ----------------------
plt.legend(prop={'size': 20, 'weight': 'bold'})

# 坐标轴边框加粗
ax = plt.gca()
for spine in ax.spines.values():
    spine.set_linewidth(2.0)

# 坐标轴刻度 & 标签
plt.xticks(fontsize=25, fontweight='bold')
plt.yticks(fontsize=25, fontweight='bold')
plt.xlabel('t-SNE Dimension 1', fontsize=25, fontweight='bold')
plt.ylabel('t-SNE Dimension 2', fontsize=25, fontweight='bold')

# 标题
plt.title('smartfall', fontsize=25, fontweight='bold')

# 布局 + 显示 + 保存
plt.tight_layout()
plt.show()
plt.savefig('./smartfall_visualization.png', dpi=600)
plt.close()