# encoding=utf-8
import matplotlib.pyplot as plt
# matplotlib.use('Agg')
from models.spike import *
from trainer import *
import torch
import torch.nn as nn
import argparse
from datetime import datetime
import numpy as np
import os
from copy import deepcopy
import fitlog
from utils import tsne, mds, _logger, hook_layers
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
# from fvcore.nn import FlopCountAnalysis, parameter_count_table
from thop import profile
import torch
from thop import profile





def print_model_flops(model, sample_input, device):
    model.eval()
    model = model.to(device)
    sample_input = sample_input.to(device)
    with torch.no_grad():
        macs, params = profile(model, inputs=(sample_input,), verbose=False)
        # macs是乘加操作数（Multiply–accumulate operations）
        # 通常FLOPs = 2 * MACs，但这里我们直接显示MACs，或乘以2得到FLOPs
        print("\nModel Parameters: {:.2f}M".format(params / 1e6))
        print("Total MACs: {:.2f}M".format(macs / 1e6))
        print("Total FLOPs: {:.2f}M".format(macs * 2 / 1e6))  # 如果需要FLOPs，macs乘以2


from thop import profile
# from torchstat import stat
# stat(model.to(device),())
#import snntorch.functional as SF
# fitlog.debug()

parser = argparse.ArgumentParser(description='argument setting of network')
parser.add_argument('--cuda', default=0, type=int, help='cuda device ID，0/1')
parser.add_argument('--rep', default=1, type=int, help='repeats for multiple runs')
# hyperparameter
parser.add_argument('--batch_size', type=int, default=64, help='batch size of training')
parser.add_argument('--n_epoch', type=int, default=100, help='number of training epochs')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--lr_cls', type=float, default=1e-3, help='learning rate for linear classifier')

# dataset
parser.add_argument('--dataset', type=str, default='hhar', choices=[ 'ucihar', 'shar', 'hhar'],
                    help='name of dataset')
parser.add_argument('--n_feature', type=int, default=77, help='name of feature dimension')
parser.add_argument('--len_sw', type=int, default=30, help='length of sliding window')
parser.add_argument('--n_class', type=int, default=18, help='number of class')
parser.add_argument('--cases', type=str, default='random', choices=['random', 'subject', 'subject_large',
                                                                    'cross_device',
                                                                    'joint_device'], help='name of scenarios')
parser.add_argument('--split_ratio', type=float, default=0.2, help='split ratio of test/val: train(0.64), val(0.16), '
                                                                   'test(0.2)')
parser.add_argument('--target_domain', type=str, default='0', help='the target domain, [0 to 29] for ucihar, '
                                                                   '[1,2,3,5,6,9,11,13,14,15,16,17,19,20,21,'
                                                                   '22,23,24,25,29] for shar, [a-i] for hhar')

# backbone model
parser.add_argument('--backbone', type=str, default='SFCN', choices=['DCL', 'FCN', 'LSTM', 'Transformer',
                                                                    'SFCN', 'SDCL'], help='name of framework')

# log
parser.add_argument('--logdir', type=str, default='log/', help='log directory')

# AE & CNN_AE
parser.add_argument('--lambda1', type=float, default=1.0,
                    help='weight for reconstruction loss when backbone in [AE, CNN_AE]')

# hhar
parser.add_argument('--device', type=str, default='Phones', choices=['Phones', 'Watch'],
                    help='data of which device to use (random case);'
                         ' data of which device to be used as training data (cross-device case,'
                         ' data from the other device as test data)')

# spike
parser.add_argument('--tau', type=float, default=0.5, help='decay for LIF')
parser.add_argument('--thresh', type=float, default=1.0, help='threshold for LIF')
parser.add_argument('--eval', action='store_true', help='Evaluation model')

# create directory for saving and plots
global plot_dir_name
plot_dir_name = 'plot/'
if not os.path.exists(plot_dir_name):
    os.makedirs(plot_dir_name)

import torch
import torch.nn as nn
import gc
import psutil
import time
#计算CPU内存占用
def benchmark_memory_usage(args, model_name_list, batch_sizes):
    DEVICE = torch.device('cuda:' + str(args.cuda) if torch.cuda.is_available() else 'cpu')
    results = []


    for model_name in model_name_list:
        for bs in batch_sizes:
            # 清除显存
            torch.cuda.empty_cache()
            gc.collect()

            args.batch_size = bs
            args.backbone = model_name

            # 初始化模型
            if args.backbone == 'FCN':
                model = FCN(n_channels=args.n_feature, n_classes=args.n_class, backbone=False)
            elif args.backbone == 'SFCN':
                model = SFCN(n_channels=args.n_feature, n_classes=args.n_class, backbone=False,
                             tau=args.tau, thresh=args.thresh)
            elif args.backbone == 'DCL':
                model = DeepConvLSTM(n_channels=args.n_feature, n_classes=args.n_class,
                                     conv_kernels=64, kernel_size=5, LSTM_units=128, backbone=False)
            elif args.backbone == 'SDCL':
                model = SDCL(args,n_channels=args.n_feature, n_classes=args.n_class,
                             conv_kernels=64, kernel_size=5, LSTM_units=128, backbone=False,
                             tau=args.tau, thresh=args.thresh)
            elif args.backbone == 'LSTM':
                model = LSTM(n_channels=args.n_feature, n_classes=args.n_class, LSTM_units=128, backbone=False)
            elif args.backbone == 'Transformer':
                model = Transformer(n_channels=args.n_feature, len_sw=args.len_sw, n_classes=args.n_class,
                                    dim=128, depth=4, heads=4, mlp_dim=64, dropout=0.1, backbone=False)
            else:
                print(f"Model {args.backbone} not supported.")
                continue

            model = model.to(DEVICE)
            model.eval()

            # 构造 dummy 输入
            # 构造 dummy input（确保 shape 是 [batch_size, channels, seq_len]）
            # 如果你的参数传反了（比如 feature 和滑窗），也能自动调整
            # 对每种模型指定期望的输入格式
            if model_name in ['FCN', 'DCL', 'SDCL', 'SFCN']:
                dummy_input = torch.randn(bs, args.n_feature, args.len_sw).to(DEVICE)  # [bs, channels, seq_len]
            elif model_name in ['LSTM', 'Transformer']:
                dummy_input = torch.randn(bs, args.len_sw, args.n_feature).to(DEVICE)  # [bs, seq_len, channels]

            print(f"[Input] dummy_input shape: {dummy_input.shape}")  # 应该是 [bs, 3, 151]

            # 同步 CUDA 计时器，记录前后内存
            torch.cuda.reset_peak_memory_stats(DEVICE)
            with torch.no_grad():
                output = model(dummy_input)

            peak_mem = torch.cuda.max_memory_allocated(DEVICE) / (1024 * 1024)  # MB

            print(f"Model: {model_name}, Batch Size: {bs}, Peak GPU Memory: {peak_mem:.2f} MB")
            results.append((model_name, bs, peak_mem))

            del model
            del dummy_input
            torch.cuda.empty_cache()
            gc.collect()


    return results


def train(args, train_loaders, val_loader, model, DEVICE, optimizer, criterion):

    # dummy_input = torch.randn(1, 9, 77).float().to(DEVICE)
    #
    # print("dummy_input shape:", dummy_input.shape)  # 应该是 (1, 77, 30)
    # print("model first conv in_channels:", model.conv1.in_channels)

    # print_model_flops(model, dummy_input, DEVICE)

    min_val_loss = 0
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Total_Number of params: {} |Trainable_num of params: {}'.format(total_num, trainable_num))
    dummy_input = torch.randn(64, 151, 3).to(DEVICE)
    flops, params = profile(model, inputs=(dummy_input,))
    print(f"真实的FLOPs: {flops / 1e9} G, 真实的Params: {params / 1e6} M")
    acc_epoch_list = []
    val_acc_epoch_list = []
    for epoch in range(args.n_epoch):
        logger.debug(f'\nEpoch : {epoch}')

        train_loss = 0
        n_batches = 0
        total = 0
        correct = 0
        model.train()
        for loader_idx, train_loader in enumerate(train_loaders):
            for idx, (sample, target, domain) in enumerate(train_loader):
                #optimizer.zero_grad() #
                n_batches += 1
                sample, target = sample.to(DEVICE).float(), target.to(DEVICE).long()
                if epoch == 0 and loader_idx == 0 and idx == 0:
                    print("真实训练输入 tensor 形状:", sample.shape)
                if args.backbone[-2:] == 'AE':
                    out, x_decoded = model(sample)
                else:
                    out, _ = model(sample)
                loss = criterion(out, target)
                if args.backbone[-2:] == 'AE':
                    # print(loss.item(), nn.MSELoss()(sample, x_decoded).item())
                    loss = loss + nn.MSELoss()(sample, x_decoded) * args.lambda1
                train_loss = train_loss + loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                _, predicted = torch.max(out.data, 1)
                #我原来的代码
                # total += target.size(0)
                # correct += (predicted == target).sum()
                #我原来的代码
                total += target.size(0)
                correct += (predicted == target).sum()
                if idx == 0:
                    y_true_train = target.detach().cpu()
                    y_pred_train = predicted.detach().cpu()
                else:
                    y_true_train = torch.cat((y_true_train, target.detach().cpu()), dim=0)
                    y_pred_train = torch.cat((y_pred_train, predicted.detach().cpu()), dim=0)

        acc_train = float(correct) * 100.0 / total
        fitlog.add_loss(train_loss / n_batches, name="Train Loss", step=epoch)
        fitlog.add_metric({"dev": {"Train Acc": acc_train}}, step=epoch)
        acc_epoch_list += [round(acc_train, 2)]

        logger.debug(f'Train Loss     : {train_loss / n_batches:.4f}\t | \tTrain Accuracy     : {acc_train:2.4f}\n')

        if val_loader is None:
            best_model = deepcopy(model.state_dict())
            model_dir = save_dir + args.model_name + '.pt'
            print('Saving models to {}'.format(model_dir))
            torch.save({'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()},
                       model_dir)
        else:
            with torch.no_grad():
                model.eval()
                val_loss = 0
                n_batches = 0
                total = 0
                correct = 0
                for idx, (sample, target, domain) in enumerate(val_loader):
                    n_batches += 1
                    sample, target = sample.to(DEVICE).float(), target.to(DEVICE).long()
                    if args.backbone[-2:] == 'AE':
                        out, x_decoded = model(sample)
                    else:
                        out, _ = model(sample)
                    loss = criterion(out, target)
                    if args.backbone[-2:] == 'AE':
                        loss += nn.MSELoss()(sample, x_decoded) * args.lambda1
                    val_loss += loss.item()
                    _, predicted = torch.max(out.data, 1)
                    #我原来的代码
                    # total += target.size(0)
                    # correct += (predicted == target).sum()
                    #我原来的代码
                    total += target.size(0)
                    correct += (predicted == target).sum()
                    if idx == 0:
                        y_true_val = target.detach().cpu()
                        y_pred_val = predicted.detach().cpu()
                    else:
                        y_true_val = torch.cat((y_true_val, target.detach().cpu()), dim=0)
                        y_pred_val = torch.cat((y_pred_val, predicted.detach().cpu()), dim=0)

                acc_val = float(correct) * 100.0 / total
                fitlog.add_loss(val_loss / n_batches, name="Val Loss", step=epoch)
                fitlog.add_metric({"dev": {"Val Acc": acc_val}}, step=epoch)
                logger.debug(f'Val Loss     : {val_loss / n_batches:.4f}\t | \tVal Accuracy     : {acc_val:2.4f}\n')
                from sklearn.metrics import classification_report

                precision = precision_score(y_true_val, y_pred_val, average='macro', zero_division=0)
                recall = recall_score(y_true_val, y_pred_val, average='macro', zero_division=0)
                f1 = f1_score(y_true_val, y_pred_val, average='macro', zero_division=0)

                print(f"Precision: {precision:.4f} | Recall: {recall:.4f} | F1 Score: {f1:.4f}")
                fitlog.add_metric({"dev": {
                    "Precision": float(precision),
                    "Recall": float(recall),
                    "F1 Score": float(f1)
                }}, step=epoch)



                val_acc_epoch_list += [round(acc_val, 2)]

                if epoch == args.n_epoch - 1:
                    cm = confusion_matrix(y_true_val, y_pred_val)
                    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
                    disp.plot(cmap=plt.cm.Blues)
                    plt.title("Confusion Matrix")
                    plt.savefig("confusion_matrix.png")  # 保存图像
                    plt.show()  # 展示图像

                if acc_val >= min_val_loss:
                    min_val_loss = acc_val
                    best_model = deepcopy(model.state_dict())
                    print('update')
                    model_dir = save_dir + args.model_name + '.pt'
                    print('Saving models to {}'.format(model_dir))
                    torch.save({'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()},
                               model_dir)

    return best_model


def test(test_loader, model, DEVICE, criterion, plt=False):
    with torch.no_grad():
        model.eval()
        total_loss = 0
        n_batches = 0
        total = 0
        correct = 0
        feats = None
        prds = None
        trgs = None
        confusion_matrix = torch.zeros(args.n_class, args.n_class)
        for idx, (sample, target, domain) in enumerate(test_loader):
            n_batches += 1
            sample, target = sample.to(DEVICE).float(), target.to(DEVICE).long()
            out, features = model(sample)
            loss = criterion(out, target)
            total_loss += loss.item()
            _, predicted = torch.max(out.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum()
            if prds is None:
                prds = predicted
                trgs = target
                feats = features[:, :]
            else:
                prds = torch.cat((prds, predicted))
                trgs = torch.cat((trgs, target))
                feats = torch.cat((feats, features), 0)

            trgs = torch.cat((trgs, target))
            feats = torch.cat((feats, features), 0)

        acc_test = float(correct) * 100.0 / total

    fitlog.add_best_metric({"dev": {"Test Loss": total_loss / n_batches}})
    fitlog.add_best_metric({"dev": {"Test Acc": acc_test}})

    logger.debug(f'Test Loss     : {total_loss / n_batches:.4f}\t | \tTest Accuracy     : {acc_test:2.4f}\n')
    for t, p in zip(trgs.view(-1), prds.view(-1)):
        confusion_matrix[t.long(), p.long()] += 1
    logger.debug(confusion_matrix)
    logger.debug(confusion_matrix.diag() / confusion_matrix.sum(1))
    fitlog.add_hyper(confusion_matrix, name='conf_mat')
    if plt == True:
        tsne(feats, trgs, domain=None, save_dir=plot_dir_name + args.model_name + '_tsne.png')
        mds(feats, trgs, domain=None, save_dir=plot_dir_name + args.model_name + 'mds.png')
        sns_plot = sns.heatmap(confusion_matrix, cmap='Blues', annot=True)
        sns_plot.get_figure().savefig(plot_dir_name + args.model_name + '_confmatrix.png')
    return acc_test


def seed_all(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def lbl_to_spike(prediction):
    N = len(prediction)
    detections = np.zeros(N)
    for i in range(1, N):
        if (prediction[i] != prediction[i-1]):
            detections[i] = prediction[i]+1
    return detections

def calculate_stats(prediction, lbl, tol):
    decisions = lbl_to_spike(prediction)
    labs = lbl_to_spike(lbl)

    lbl_indices = np.nonzero(labs)
    lbl_indices = np.array(lbl_indices).flatten()

    dist = np.zeros((len(lbl_indices), 6))
    for i in range(len(lbl_indices)):
        index = lbl_indices[i]
        lab = int(labs[index])
        dec_indices = np.array(np.nonzero((decisions-lab) == 0)).flatten()  #indices where decisions == lab
        if len(dec_indices) == 0:
            dist[i, lab - 1] = 250
            continue
        j = np.argmin(np.abs(dec_indices - index))  # j is closest val in dec_indices to index
        dist[i, lab-1] = abs(dec_indices[j]-index)
        if (dist[i, lab-1] <= tol):
            decisions[dec_indices[j]] = 0 # mark as handled

    mean_error = np.mean(dist, axis=0)
    TP = np.sum(dist <= tol, axis=0)
    FN = np.sum(dist > tol, axis=0)

    FP = np.zeros(6)
    for i in decisions[(decisions > 0)]:
        FP[int(i-1)] += 1

    return mean_error, TP, FN, FP


if __name__ == '__main__':

    args = parser.parse_args()
    from argparse import Namespace

    print(f"[DEBUG] n_feature = {args.n_feature}, len_sw = {args.len_sw}")

    DEVICE = torch.device('cuda:' + str(args.cuda) if torch.cuda.is_available() else 'cpu')
    print('device:', DEVICE, 'dataset:', args.dataset)
    args.input_shape = (args.n_feature, args.len_sw)  # e.g., (77, 30)
    acc_list = []
    # log
    args.model_name = args.backbone + '_' + args.dataset + '_lr' + str(args.lr) + '_bs' + str(
        args.batch_size) + '_sw' + str(args.len_sw)

    if os.path.isdir(args.logdir) == False:
        os.makedirs(args.logdir)
    log_file_name = os.path.join(args.logdir, args.model_name + f".log")
    logger = _logger(log_file_name)
    logger.debug(args)

    # fitlog
    fitlog.set_log_dir(args.logdir)
    fitlog.add_hyper(args)
    fitlog.add_hyper_in_file(__file__)

    training_start = datetime.now()

    for r in range(args.rep):

        # fix random seed for reproduction
        seed_all(seed=1000 + r)
        train_loaders, val_loader, test_loader = setup_dataloaders(args)

        snn_params = {"tau": args.tau, "thresh": args.thresh}


        if not args.eval:

            if args.backbone == 'FCN':
                model = FCN(n_channels=args.n_feature, n_classes=args.n_class, backbone=False)
            elif args.backbone == 'SFCN':
                model = SFCN(n_channels=args.n_feature, n_classes=args.n_class, backbone=False, **snn_params)
            elif args.backbone == 'DCL':
                model = DeepConvLSTM(n_channels=args.n_feature, n_classes=args.n_class, conv_kernels=64, kernel_size=5,
                                     LSTM_units=128, backbone=False)
            elif args.backbone == 'SDCL':
                model = SDCL(n_channels=args.n_feature, n_classes=args.n_class, conv_kernels=64, kernel_size=5,
                             LSTM_units=128, backbone=False, **snn_params)
            elif args.backbone == 'LSTM':
                model = LSTM(n_channels=args.n_feature, n_classes=args.n_class, LSTM_units=128, backbone=False)
            elif args.backbone == 'Transformer':
                model = Transformer(n_channels=args.n_feature, len_sw=args.len_sw, n_classes=args.n_class, dim=128,
                                    depth=4, heads=4, mlp_dim=64, dropout=0.1, backbone=False)
            else:
                raise NotImplementedError

            model = model.to(DEVICE)
            # stat(model.to(DEVICE), (3,64,151))

            save_dir = 'results/'
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            criterion = nn.CrossEntropyLoss()
            #criterion = SF.mse_count_loss(correct_rate=1, incorrect_rate=0.3)

            parameters = model.parameters()
            optimizer = torch.optim.Adam(parameters, args.lr)
            # 初始化余弦退火学习率调度器，T_max 是最大训练轮次
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
            train_loss_list = []
            test_loss_list = []

            best_model = train(args, train_loaders, val_loader, model, DEVICE, optimizer, criterion)

        else:
            criterion = nn.CrossEntropyLoss()
            save_dir = 'results/'
            best_model = torch.load(save_dir + args.model_name + '.pt')['model_state_dict']

        if args.backbone == 'FCN':
            model_test = FCN(n_channels=args.n_feature, n_classes=args.n_class, backbone=False)
        elif args.backbone == 'SFCN':
            model_test = SFCN(n_channels=args.n_feature, n_classes=args.n_class, backbone=False, **snn_params)
        elif args.backbone == 'DCL':
            model_test = DeepConvLSTM(n_channels=args.n_feature, n_classes=args.n_class, conv_kernels=64, kernel_size=5,
                                      LSTM_units=128, backbone=False)
        elif args.backbone == 'SDCL':
            model_test = SDCL(args,
                              n_channels=args.n_feature, n_classes=args.n_class, conv_kernels=64, kernel_size=5,
                              LSTM_units=128, backbone=False, **snn_params)
        elif args.backbone == 'LSTM':
            model_test = LSTM(n_channels=args.n_feature, n_classes=args.n_class, LSTM_units=128, backbone=False)
        elif args.backbone == 'Transformer':
            model_test = Transformer(n_channels=args.n_feature, len_sw=args.len_sw, n_classes=args.n_class, dim=128,
                                     depth=4, heads=4, mlp_dim=64, dropout=0.1, backbone=False)
        else:
            raise NotImplementedError

        model_test.load_state_dict(best_model)
        model_test = model_test.to(DEVICE)

        avgmeter = hook_layers(model_test)

        test_loss = test(test_loader, model_test, DEVICE, criterion, plt=False)
        scheduler.step() #更新学习率
        acc_list.append(test_loss)

        print("Fire Rate: {}".format(avgmeter.avg()))

    training_end = datetime.now()
    training_time = training_end - training_start
    logger.debug(f"Training time is : {training_time}")

    a = np.array(acc_list)
    print('Final Accuracy: {}, Std: {}'.format(np.mean(a), np.std(a)))
    if args.dataset == 'shar':  # 只在SHAR数据集上测试
        model_list = ['FCN', 'DCL', 'SDCL', 'LSTM', 'SFCN']
        batch_sizes = [16, 32, 64, 128, 256]

