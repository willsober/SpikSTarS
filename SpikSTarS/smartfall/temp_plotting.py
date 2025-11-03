from model import SNNModel0HLayers
from dataloader import SisFallDataset
from torch.utils.data import DataLoader
from utils import *
import snntorch as snn
from snntorch import functional as SF
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

import pandas as pd
import matplotlib.pyplot as plt

import tqdm
import sys
import os
import numpy as np

def accuracy_modified(spikes, label):
    # print(spikes.shape)
    rate_spike = torch.sum(spikes, dim=0)/len(spikes)
    # print(f'Num of Spikes: {torch.sum(spikes, dim=0)}')
    # print(f'Rate of Spike: {rate_spike}')

    spike_count_rate = []
    for i in range(0, len(spikes)):
        spike_count_rate.append(torch.sum((spikes[0:i]), axis=0)/i)
    spike_count_rate[0] = torch.tensor([[0.,0.]], device='mps:0')

    # print(f'Mean Del Spike Rate: {spike_count_rate}')
    spike_count_rate_sum = sum(spike_count_rate)
    # print(spike_count_rate_sum)

    

    pred = torch.argmax(spike_count_rate_sum)
    # print(pred)
    # print(f'Label: {label}')
    # print('--')
    accuracy = np.mean((label==pred).detach().cpu().numpy())
    return accuracy



def accuracy_modified2(spikes, label):
    # print(spikes.shape)
    num_steps, _, num_classes = spikes.shape
    # print(f'Num of Spikes: {torch.sum(spikes, dim=0)}')
    # print(f'Rate of Spike: {rate_spike}')

    spike_count = torch.zeros((num_steps, num_classes))
    for i in range(5, len(spikes)):
        # spike_count.append(torch.sum((spikes[0:i]), axis=0)/i)
        spike_count[i] = torch.sum((spikes[i-5:i]), axis=0)
    # spike_count[0:5] = torch.tensor([[0.,0.]], device='mps:0')
    # print(spike_count)
    # print(f'Mean Del Spike Rate: {spike_count_rate}')
    spike_count_sum = torch.sum(spike_count, dim=0)
    # print(spike_count_sum)

    

    pred = torch.argmax(spike_count_sum)
    # print(pred)
    # print(f'Label: {label}')
    # print('--')
    accuracy = np.mean((label==pred).detach().cpu().numpy())
    # print(accuracy)
    return accuracy


def accuracy_modified3(spikes, label):
    # print(spikes.shape)
    num_steps, _, num_classes = spikes.shape
    # print(f'Num of Spikes: {torch.sum(spikes, dim=0)}')
    # print(f'Rate of Spike: {rate_spike}')

    spike_count = torch.zeros((num_steps, num_classes))
    for i in range(20, len(spikes)):
        # spike_count.append(torch.sum((spikes[0:i]), axis=0)/i)
        spike_count[i] = torch.sum((spikes[i-20:i:5]), axis=0)
    # spike_count[0:5] = torch.tensor([[0.,0.]], device='mps:0')
    # print(spike_count)
    # print(f'Mean Del Spike Rate: {spike_count_rate}')
    spike_count_sum = torch.sum(spike_count, dim=0)
    # print(spike_count_sum)

    

    pred = torch.argmax(spike_count_sum)
    # print(pred)
    # print(f'Label: {label}')
    # print('--')
    accuracy = np.mean((label==pred).detach().cpu().numpy())
    # print(accuracy)
    return accuracy



torch.manual_seed(4)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

num_steps = 50
input_size = 2
hidden_size = 3
output_size = 2

batch_size = 1

test_dataset = SisFallDataset('/Users/archit/Documents/Projects/LLLF/dataset/sis_fall/time_window_500ms_sliding_50ms/test')
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=False)

# List of model directories
# model_directories = [
#     './saves/checkpoint_sliding_fd.pt',
#     './saves/checkpoint_sliding_fd_ns50_multiloss.pt',
#     # './saves/checkpoint_sliding_fd_ns50_squareweightedcountloss.pt',
#     # './saves/checkpoint_sliding_fd_ns50_sqrtweightedcountloss.pt',
#     # './saves/checkpoint_sliding_fd_ns50_cecountloss.pt',
#     './saves/checkpoint_sliding_fd_ns50_rev-linear-loss.pt',
#     './saves/checkpoint_sliding_fd_ns50_linear-dec-loss.pt'
# ]
checkpoint_path = './saves/snn_model_500ms_50ep_lin_wt_delay-0.pt'

# model_names = [
#     'MSE Count Loss',
#     'Linear Weight MSE Count Loss',
#     # 'Quadratic Weight MSE Count Loss',
#     # 'SQRT Weighted MSE Count Loss',
#     # 'Cross Entropy Loss',
#     'Reversed Linear Weighted Loss',
#     'Linearly Decreasing Weighted Loss'
# ]

colors = [
    'red',
    'blue',
    'green',
    'purple',
    # 'black',
]

# Initialize a dictionary to store accuracy logs for each model
acc_logs = {}


model = SNNModel0HLayers(time_steps=num_steps, input_features=6)


if os.path.exists(checkpoint_path):
    print(f"Loading checkpoint from {checkpoint_path}")
    model.load_state_dict(torch.load(checkpoint_path, weights_only=True))

for name, param in model.named_parameters():
    print(name)
    print(param.data)

model = model.to(device)
model.eval()

mod_acc_log = []
acc_log = []
spk_log = []
label_log = []

with torch.no_grad():
    for _, (inputs, label) in enumerate(test_loader):

        spikes_up_input, spikes_down_input = lc_sampling(inputs)
        inputs = torch.concat((spikes_up_input, spikes_down_input), dim=1)
        inputs = inputs.to(device)
        label = label.to(device)
        target = torch.argmax(label, dim=1)

        spk = model(inputs)


        acc_per_step = []
        mod_acc_per_step = []
        spk_per_step = []

        del_spike_count = []
        
        for i in range(1, len(spk) + 1):
            spikes = spk[0:i]
            spk_count = torch.sum(spikes, dim=0)
            spk_per_step.append(torch.squeeze(spk_count).tolist())

            acc = SF.accuracy_rate(spikes, target)
            acc_per_step.append(acc.item())

            # Modified Accuracy
            # print(f'Step: {i}')
            mod_acc = accuracy_modified3(spikes, target)
            mod_acc_per_step.append(mod_acc.item())
            # del_spike = torch.sum(spikes, dim=0) - torch.sum(spikes[:-1], dim=0)
            # del_spike_count.append(del_spike)
            # print(torch.argmax(sum(del_spike_count)))
        # print(label)
        # print('-----')

        # Plotting Every Accuracy per step
        # plt.plot(acc_per_step, label='Accuracy')
        # plt.plot(mod_acc_per_step, label='Modified Accuracy')
        # plt.xlabel('Time Steps')
        # plt.ylabel('Accuracy')
        # plt.title('Accuracy vs Time Steps for Different Models')
        # plt.legend()
        # plt.show()
        # plt.clf()


        acc_log.append(acc_per_step)
        mod_acc_log.append(mod_acc_per_step)
        spk_log.append(spk_per_step)
        label_log.append(target.item())

        # print(spk_log[-1][-1])
        # print(label.item())

# Convert accuracy log to numpy array and compute mean accuracy per step
acc_log = np.asarray(acc_log)
mean_acc = np.mean(acc_log, axis=0)

mod_acc_log = np.asarray(mod_acc_log)
mean_mod_acc = np.mean(mod_acc_log, axis=0)

spk_log = np.asarray(spk_log)
mean_spk = np.mean
label_log = np.asarray(label_log)
# print(spk_log.shape)
num_correct_spikes = []
num_incorrect_spikes = []
for idx in range(len(label_log)):
    label = int(label_log[idx])
    # spikes = spk_log[idx]
    num_correct_spikes.append(spk_log[idx][:,label])
    num_incorrect_spikes.append(spk_log[idx][:,1-label])
    
num_correct_spikes = np.asarray(num_correct_spikes)
num_incorrect_spikes = np.asarray(num_incorrect_spikes)

for i in range(len(num_correct_spikes)):
    plt.plot(num_correct_spikes[i], linewidth=1, color='blue', label='Correct Class')
    plt.plot(num_incorrect_spikes[i], linewidth=1, color='red', label='Incorrect Class')

    plt.xlabel('Time Steps')
    plt.ylabel('Number of Spikes')
    plt.title('Number of Spikes vs Time Steps for Different Classes')
    plt.legend()
    plt.show()
    plt.clf()

mean_num_correct_spikes = np.mean(num_correct_spikes, axis=0)
mean_num_incorrect_spikes = np.mean(num_incorrect_spikes, axis=0)


plt.plot(mean_num_correct_spikes, label='Correct Class')
plt.plot(mean_num_incorrect_spikes, label='Incorrect Class')
plt.xlabel('Time Steps')
plt.ylabel('Number of Spikes')
plt.title('Number of Spikes vs Time Steps for Different Classes')
plt.legend()
plt.show()
plt.clf()

mean_num_correct_spikes_increase_rate = []
mean_num_incorrect_spikes_increase_rate = []

mean_num_correct_spikes_increase_rate.append(0)
mean_num_incorrect_spikes_increase_rate.append(0)
for i in range(1, len(mean_num_correct_spikes)):
    mean_num_correct_spikes_increase_rate.append(mean_num_correct_spikes[i] - mean_num_correct_spikes[i-1])
    mean_num_incorrect_spikes_increase_rate.append(mean_num_incorrect_spikes[i] - mean_num_incorrect_spikes[i-1])

mean_num_correct_spikes_increase_rate = np.asarray(mean_num_correct_spikes_increase_rate)
mean_num_incorrect_spikes_increase_rate = np.asarray(mean_num_incorrect_spikes_increase_rate)

# for i in range(1, len(mean_num_correct_spikes_increase_rate)):
#     mean_num_correct_spikes_increase_rate[i] = sum(mean_num_correct_spikes_increase_rate[0:i])/i
#     mean_num_incorrect_spikes_increase_rate[i] = sum(mean_num_incorrect_spikes_increase_rate[0:i])/i

plt.plot(mean_num_correct_spikes_increase_rate, label='Correct Spikes')
plt.plot(mean_num_incorrect_spikes_increase_rate, label='Incorrect Spikes')
plt.xlabel('Time Steps')
plt.ylabel('Rate of Increase')
plt.title('Rate of Increase of Num Spikes vs Time Steps for Different Classes')
plt.legend()
plt.show()
plt.clf()



plt.plot(mean_acc, label='Accuracy')
plt.plot(mean_mod_acc, label='Modified Accuracy')
plt.xlabel('Time Steps')
plt.ylabel('Accuracy')
plt.title('Accuracy vs Time Steps for Different Models')
plt.legend()
plt.grid()
plt.show()
plt.clf()




