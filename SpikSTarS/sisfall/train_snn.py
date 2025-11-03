import torch
import torch.nn as nn
import snntorch as snn
from snntorch import functional as SF
from dataloader import SisFallDataset
from torch.utils.data import DataLoader
from utils import *
import torch.optim.lr_scheduler as lr_scheduler
from criterion import *
from model import SNNModel0HLayers
import tqdm
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np

# from fvcore.nn import FlopCountAnalysis, parameter_count_table

# def print_model_flops(model, sample_input, device):
#     model.eval()
#     model.to(device)
#     sample_input = sample_input.to(device)
#     with torch.no_grad():
#         flops = FlopCountAnalysis(model, sample_input)
#         print("Model Parameters:")
#         print(parameter_count_table(model))
#         print(f"FLOPs: {flops.total():,} ({flops.total() / 1e6:.2f} MFLOPs)")

torch.manual_seed(41) #0-9
#0, 41, 1234, 1984, 111, 2718, 666, 2468
class AvgrageMeter(object):

  def __init__(self):
    self.reset()

  def reset(self):
    self.avg = 0
    self.sum = 0
    self.cnt = 0

  def update(self, val, n=1):
    self.sum += val * n
    self.cnt += n
    self.avg = self.sum / self.cnt


def accuracy(output, target, topk=(1,)):
  maxk = max(topk)
  batch_size = target.size(0)

  _, pred = output.topk(maxk, 1, True, True)
  pred = pred.t()
  correct = pred.eq(target.view(1, -1).expand_as(pred))

  res = []
  for k in topk:
    correct_k = correct[:k].reshape(-1).float().sum(0)
    res.append(correct_k.mul_(100.0/batch_size))
  return res



device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

num_steps = 25
batch_size = 4
train_dataset = SisFallDataset('/tmp/pycharm_project_582/dataset/sis_fall/time_window_500ms_sliding_50ms/train')
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
test_dataset = SisFallDataset('/tmp/pycharm_project_582/dataset/sis_fall/time_window_500ms_sliding_50ms/test')
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=True)


snn_model = SNNModel0HLayers(time_steps=num_steps, input_features=6)
snn_model = snn_model.to(device)
num_epochs = 150
optimizer = torch.optim.Adam(params=snn_model.parameters(), lr=0.01,betas=(0.9, 0.999)) #0.01效果最好
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)
# scheduler = CosineAnnealingLR(optimizer, T_max=10, eta_min=0.0001)
# total_num = sum(p.numel() for p in snn_model.parameters())
# trainable_num = sum(p.numel() for p in snn_model.parameters() if p.requires_grad)
# print('Total_Number of params: {} |Trainable_num of params: {}'.format(total_num, trainable_num))
# Estimate FLOPs
# dummy_input = torch.rand((1, 12, num_steps), dtype=torch.float32)
# print_model_flops(snn_model, dummy_input, device)

loss_function = SF.mse_count_loss(correct_rate=0.7, incorrect_rate=0.3)
# loss_function = nn.CrossEntropyLoss()
# # 初始化余弦退火学习率调度器
#

checkpoint_path = './saves/snn_model_500ms_30ep_mse_count_loss_quick_encoding_seed41.pt'  

#snn_model_500ms_30ep_linear_weighted_quick_encoding_seed0
#snn_model_500ms_50ep_mse_count_loss_lc_sampling_seed0
# print(">>> Estimating FLOPs ...")
# dummy_input = torch.rand((1, 12, num_steps), dtype=torch.float32)
# print_model_flops(snn_model, dummy_input, device)
# print(">>> FLOPs estimation done.")

import utils
def get_predictions(spk):
    # If spk shape is [T, B, C], convert to [B, C, T]
    if spk.shape[0] == num_steps:
        spk = spk.permute(1, 2, 0)
    elif spk.shape[2] != num_steps:
        raise ValueError(f"Unexpected spk shape: {spk.shape}")

    spk_sum = spk.sum(dim=2)
    _, pred = spk_sum.max(dim=1)
    return pred

with tqdm.trange(num_epochs) as pbar:

    #参数打印
    total_num = sum(p.numel() for p in snn_model.parameters())
    trainable_num = sum(p.numel() for p in snn_model.parameters() if p.requires_grad)
    # print('Total_Number of params: {} |Trainable_num of params: {}'.format(total_num, trainable_num))
    total_bit_synops = AvgrageMeter()
    for epoch in pbar:
        snn_model.train()
        train_acc = 0
        train_loss = 0
        val_acc = 0
        val_loss = 0
        TP = 0  # Fall - Detected
        FP = 0  # Not Fall - Detected
        TN = 0  # Not Fall - Not Detected
        FN = 0  # Fall - Not Detected
        all_preds = []
        all_targets = []

        for _, (inputs, label) in enumerate(train_loader):

            #Quick Encoding
            inputs = quick_spikes_encoding(inputs, 25)
            inputs = torch.cat((inputs, torch.flip(inputs, dims=[2])), dim=1)

            # LC Sampling
            # spikes_up_input, spikes_down_input = lc_sampling(inputs)
            # spikes_up_input = time_slot_accumulation(spikes_up_input, sampling_freq=100, subsampling_freq=50)
            # spikes_down_input = time_slot_accumulation(spikes_down_input, sampling_freq=100, subsampling_freq=50)
            # inputs = torch.concat((spikes_up_input, spikes_down_input), dim=1)

            inputs = inputs.to(device)
            label = label.to(device)
            target = torch.argmax(label, dim=1)
            # print(inputs.shape)
            spk = snn_model(inputs)  # forward-pass
            pred = get_predictions(spk)

            loss_val = loss_function(spk, target)
            train_loss += loss_val.item()

            optimizer.zero_grad()  # zero out gradients
            loss_val.backward()  # calculate gradients
            optimizer.step()  # update weights

          #get predicted class
            # Prediction



            #原来的代码
            # acc_val = SF.accuracy_rate(spk, target)
            # train_acc += acc_val

            # Accuracy
            acc_val = (pred == target).float().mean()
            train_acc += acc_val.item()
            # TP/FP/TN/FN 统计
            for t, p in zip(target, pred):
                if t == 0 and p == 0:
                    TP += 1
                elif t == 0 and p != 0:
                    FN += 1
                elif t != 0 and p == 0:
                    FP += 1
                elif t != 0 and p != 0:
                    TN += 1


           



        avg_loss = train_loss / len(train_loader)
        avg_acc = train_acc / len(train_loader)

        print(f"Epoch [{epoch + 1}/{num_epochs}] - Avg Loss: {avg_loss:.5f}, Avg Accuracy: {avg_acc:.5f}")
        # print(f"Epoch [{epoch + 1}/{num_epochs}] - Avg_val_loss: {avg_val_loss:.5f}, Avg_val_acc: {avg_val_acc:.5f}")

        snn_model.eval()
        with torch.no_grad():
            for _, (inputs, label) in enumerate(test_loader):

               #Quick Encoding
                inputs = quick_spikes_encoding(inputs, 25)
                inputs = torch.cat((inputs, torch.flip(inputs, dims=[2])), dim=1)
                # n = input.size(0)

                # LC Sampling
                # spikes_up_input, spikes_down_input = lc_sampling(inputs)
                # spikes_up_input = time_slot_accumulation(spikes_up_input, sampling_freq=100, subsampling_freq=50)
                # spikes_down_input = time_slot_accumulation(spikes_down_input, sampling_freq=100, subsampling_freq=50)
                # inputs = torch.concat((spikes_up_input, spikes_down_input), dim=1)


                inputs = inputs.to(device)
                label = label.to(device)
                target = torch.argmax(label, dim=1)

                # spk = snn_model(inputs)
                #
                # loss_val = loss_function(spk, target)
                # val_loss += loss_val.item()
                #
                # acc_val = SF.accuracy_rate(spk, target)
                # val_acc += acc_val
                #原来的代码
                spk = snn_model(inputs)
                pred = get_predictions(spk)

                loss_val = loss_function(spk, target)
                val_loss += loss_val.item()



                acc_val = (pred == target).float().mean()
                val_acc += acc_val.item()

                # 验证集统计也可加TP/FP/TN/FN（如果你想看验证集表现）
                for t, p in zip(target, pred):
                    if t == 0 and p == 0:
                        TP += 1
                    elif t == 0 and p != 0:
                        FN += 1
                    elif t != 0 and p == 0:
                        FP += 1
                    elif t != 0 and p != 0:
                        TN += 1
                all_preds.extend(pred.cpu().numpy())
                all_targets.extend(target.cpu().numpy())

        avg_val_loss = val_loss / len(test_loader)
        avg_val_acc = val_acc / len(test_loader)


        # total_bit_synops.update(model_bit_synops, n)
        print(f"Epoch [{epoch + 1}/{num_epochs}] - Avg_val_loss: {avg_val_loss:.5f}, Avg_val_acc: {avg_val_acc:.5f}")
        print("Accuracy: ", (TP + TN) / (TP + FP + TN + FN))
        if (TP + FP) > 0:
            precision = TP / (TP + FP)
        else:
            precision = 0.0  # 或者其他适当的处理方式，如设为 NaN
        print("Precision: ", precision)

        print("Recall: ", TP / (TP + FN))
        print("F1 Score: ", 2 * TP / (2 * TP + FP + FN))
        if (epoch + 1) == num_epochs:
            class_names = ["Fall", "Not Fall"]  # 或者根据你的类别自行修改
            cm = confusion_matrix(all_targets, all_preds)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
            disp.plot(cmap=plt.cm.Blues)
            plt.title(f"Confusion Matrix")
            plt.savefig("confusion_matrix.png")  # 将图像保存到本地
            plt.show()



        # total_bit_synops.update(model_bit_synops, n)
        scheduler.step(avg_val_loss)
        # scheduler.step()

        # pbar.set_postfix({
        #     'epoch': epoch + 1,
        #     'train_loss': '{0:1.5f}'.format(avg_loss),
        #     'train_acc': '{:.5f}'.format(avg_acc),
        #     'val_loss': '{0:1.5f}'.format(avg_val_loss),
        #     'val_acc': '{:.5f}'.format(avg_val_acc)
        # })

        if (epoch + 1) % 5 == 0:
            torch.save(snn_model.state_dict(), checkpoint_path)

