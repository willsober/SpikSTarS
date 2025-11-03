import torch
import torch.nn as nn
import snntorch as snn
from snntorch import functional as SF
from dataloader import SisFallDataset
from torch.utils.data import DataLoader
import numpy as np
from utils import *
import matplotlib.pyplot as plt
from model import SNNModel0HLayers
import pandas as pd

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

num_steps = 25
batch_size = 1

test_dataset = SisFallDataset('/Users/hemanthsabbella/Documents/SNN_Responsiveness/dataset/sis_fall/time_window_500ms_sliding_50ms/test')
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)


snn_model = SNNModel0HLayers(time_steps=num_steps, input_features=6)
snn_model.load_state_dict(torch.load("saves/snn_model_500ms_30ep_mse_count_loss_quick_encoding_seed41.pt", weights_only=True))
snn_model = snn_model.to(device)


acc_log = []
with torch.no_grad():
    for _, (inputs, label) in enumerate(test_loader):

        inputs = quick_spikes_encoding(inputs, 25)
        inputs = torch.cat((inputs, torch.flip(inputs, dims=[2])), dim=1)

    
        inputs = inputs.to(device)
        label = label.to(device)
        target = torch.argmax(label, dim=1)


        spk = snn_model(inputs)  # forward-pass

        acc_per_step = []
        
        for i in range(1, len(spk) + 1):
            spikes = spk[0:i]
            acc = SF.accuracy_rate(spikes, target)
            acc_per_step.append(acc.item())
        
        acc_log.append(acc_per_step)

acc_log = np.asarray(acc_log)
mean_acc = np.mean(acc_log, axis=0)
plt.plot(mean_acc, label='Quick_Encoding_Linear_Weighted_Seed1984')

file_path = 'files_4/Quick_Encoding_MSE_Count_Loss_Seed41.csv'
timesteps = np.arange(25)
data = {'Timesteps':timesteps,
        'Accuracy':mean_acc*100}
df = pd.DataFrame(data)
df.to_csv(file_path, index=False)


# snn_model = SNNModel0HLayers(time_steps=num_steps, input_features=6)
# snn_model.load_state_dict(torch.load("saves/snn_model_500ms_10ep_mse_count_loss_lc_sampling_seed1234.pt", weights_only=True))
# snn_model = snn_model.to(device)


# acc_log = []
# with torch.no_grad():
#     for _, (inputs, label) in enumerate(test_loader):
       

#         spikes_up_input, spikes_down_input = lc_sampling(inputs)
#         spikes_up_input = time_slot_accumulation(spikes_up_input, sampling_freq=100, subsampling_freq=50)
#         spikes_down_input = time_slot_accumulation(spikes_down_input, sampling_freq=100, subsampling_freq=50)
#         inputs = torch.concat((spikes_up_input, spikes_down_input), dim=1)
#         inputs = inputs.to(device)
#         label = label.to(device)
#         target = torch.argmax(label, dim=1)


#         spk = snn_model(inputs)  # forward-pass

#         acc_per_step = []
        
#         for i in range(1, len(spk) + 1):
#             spikes = spk[0:i]
#             acc = SF.accuracy_rate(spikes, target)
#             acc_per_step.append(acc.item())
        
#         acc_log.append(acc_per_step)

# acc_log = np.asarray(acc_log)
# mean_acc = np.mean(acc_log, axis=0)
# plt.plot(mean_acc, label='LC_Sampling_MSE_Count_Loss_Seed0')

# file_path = 'files/LC_Sampling_MSE_Count_Loss_Seed1234.csv'
# timesteps = np.arange(25)
# data = {'Timesteps':timesteps,
#         'Accuracy':mean_acc*100}
# df = pd.DataFrame(data)
# df.to_csv(file_path, index=False)


plt.xlabel('Time Steps')
plt.ylabel('Accuracy')
plt.title('Accuracy vs Time Steps')
plt.legend()
plt.grid()
plt.show()
plt.clf()


