import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
import pickle as cp
import scipy.io
import matplotlib.pyplot as plt
from data_preprocess.data_preprocess_utils import get_sample_weights, train_test_val_split
from data_preprocess.base_loader import base_loader


def load_domain_data(domain_idx):
    """加载特定领域的数据"""
    data_dir = './UniMiB-SHAR-main/'
    saved_filename = 'shar_domain_' + domain_idx + '_wd.data'  # "wd": with domain label

    if os.path.isfile(data_dir + saved_filename):
        data = np.load(data_dir + saved_filename, allow_pickle=True)
        X = data[0][0]
        y = data[0][1]
        d = data[0][2]
    else:
        str_folder = './UniMiB-SHAR-main/data/'
        data_all = scipy.io.loadmat(str_folder + 'acc_data.mat')
        y_id_all = scipy.io.loadmat(str_folder + 'acc_labels.mat')
        y_id_all = y_id_all['acc_labels']  # (11771, 3)

        X_all = data_all['acc_data']  # data: (11771, 453)
        y_all = y_id_all[:, 0] - 1  # to map the labels to [0, 16]
        id_all = y_id_all[:, 1]

        print(f'\nProcessing domain {domain_idx} files...\n')

        target_idx = np.where(id_all == int(domain_idx))
        X = X_all[target_idx]
        y = y_all[target_idx]

        domain_idx_map = {'1': 0, '2': 1, '3': 2, '5': 3}
        domain_idx_int = domain_idx_map[domain_idx]

        d = np.full(y.shape, domain_idx_int, dtype=int)

        print(f'\nProcessing domain {domain_idx} files | X: {X.shape}, y: {y.shape}, d: {d.shape} \n')
        obj = [(X, y, d)]
        f = open(os.path.join(data_dir, saved_filename), 'wb')
        cp.dump(obj, f, protocol=cp.HIGHEST_PROTOCOL)
        f.close()

    return X, y, d


def load_domain_data_large(domain_idx):
    """加载大规模数据集"""
    data_dir = './UniMiB-SHAR-main/'
    saved_filename = 'shar_domain_' + domain_idx + '_wd.data'

    if os.path.isfile(data_dir + saved_filename):
        data = np.load(data_dir + saved_filename, allow_pickle=True)
        X = data[0][0]
        y = data[0][1]
        d = data[0][2]
    else:
        str_folder = './UniMiB-SHAR-main/data/'
        data_all = scipy.io.loadmat(str_folder + 'acc_data.mat')
        y_id_all = scipy.io.loadmat(str_folder + 'acc_labels.mat')
        y_id_all = y_id_all['acc_labels']

        X_all = data_all['acc_data']
        y_all = y_id_all[:, 0] - 1
        id_all = y_id_all[:, 1]

        print(f'\nProcessing domain {domain_idx} files...\n')

        target_idx = np.where(id_all == int(domain_idx))
        X = X_all[target_idx]
        y = y_all[target_idx]

        domain_idx_map = {'1': 0, '2': 1, '3': 2, '5': 3, '6': 4, '9': 5,
                          '11': 6, '13': 7, '14': 8, '15': 9, '16': 10, '17': 11, '19': 12, '20': 13,
                          '21': 14, '22': 15, '23': 16, '24': 17, '25': 18, '29': 19}
        domain_idx_int = domain_idx_map[domain_idx]

        d = np.full(y.shape, domain_idx_int, dtype=int)

        print(f'\nProcessing domain {domain_idx} files | X: {X.shape}, y: {y.shape}, d: {d.shape} \n')

        obj = [(X, y, d)]
        f = open(os.path.join(data_dir, saved_filename), 'wb')
        cp.dump(obj, f, protocol=cp.HIGHEST_PROTOCOL)
        f.close()

    return X, y, d


class data_loader_shar(base_loader):
    def __init__(self, samples, labels, domains):
        super(data_loader_shar, self).__init__(samples, labels, domains)


def plot_label_distribution(y):
    """绘制标签分布图，使用 Times New Roman 字体"""
    # 设置字体为 Times New Roman，字号为 20
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 20

    unique_y, counts_y = np.unique(y, return_counts=True)

    # 绘制柱状图
    plt.bar(unique_y, counts_y)
    plt.title('Label Distribution')
    plt.xlabel('Labels')
    plt.ylabel('Frequency')

    # 修改坐标轴刻度字体大小
    plt.tick_params(axis='both', labelsize=20)

    # 显示图形
    plt.show()


def prep_domains_shar_subject(args, SLIDING_WINDOW_LEN=0, SLIDING_WINDOW_STEP=0):
    source_domain_list = ['1', '2', '3', '5']
    source_domain_list.remove(args.target_domain)

    x_win_all, y_win_all, d_win_all = np.array([]), np.array([]), np.array([])
    for source_domain in source_domain_list:
        print(f'source_domain: {source_domain}')
        x, y, d = load_domain_data(source_domain)

        x = x.reshape(-1, 151, 3)
        print(f" ..after sliding window: inputs {x.shape}, targets {y.shape}")

        x_win_all = np.concatenate((x_win_all, x), axis=0) if x_win_all.size else x
        y_win_all = np.concatenate((y_win_all, y), axis=0) if y_win_all.size else y
        d_win_all = np.concatenate((d_win_all, d), axis=0) if d_win_all.size else d

    unique_y, counts_y = np.unique(y_win_all, return_counts=True)
    print('y_train label distribution: ', dict(zip(unique_y, counts_y)))

    # 绘制标签分布图
    plot_label_distribution(y_win_all)

    weights = 100.0 / torch.Tensor(counts_y)
    print('weights of sampler: ', weights)
    weights = weights.double()

    sample_weights = get_sample_weights(y_win_all, weights)
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights=sample_weights,
                                                             num_samples=len(sample_weights), replacement=True)

    data_set = data_loader_shar(x_win_all, y_win_all, d_win_all)
    source_loader = DataLoader(data_set, batch_size=args.batch_size, shuffle=False, drop_last=True, sampler=sampler)
    print(f'source_loader batch: {len(source_loader)}')
    source_loaders = [source_loader]

    # target domain data prep
    print(f'target_domain: {args.target_domain}')
    x, y, d = load_domain_data(args.target_domain)
    x = x.reshape(-1, 151, 3)
    print(f" ..after sliding window: inputs {x.shape}, targets {y.shape}")

    data_set = data_loader_shar(x, y, d)
    target_loader = DataLoader(data_set, batch_size=args.batch_size, shuffle=False)
    print(f'target_loader batch: {len(target_loader)}')

    return source_loaders, None, target_loader


def prep_shar(args, SLIDING_WINDOW_LEN=0, SLIDING_WINDOW_STEP=0):
    if args.cases == 'subject':
        return prep_domains_shar_subject(args, SLIDING_WINDOW_LEN, SLIDING_WINDOW_STEP)
    else:
        return 'Error!\n'


# 示例调用代码
class Args:
    def __init__(self):
        self.target_domain = '1'
        self.cases = 'subject'
        self.batch_size = 32


args = Args()
source_loaders, _, target_loader = prep_shar(args)
