import torch
import torch.nn as nn
import snntorch as snn
from snntorch import functional as SF
from dataloader import SmartFallDataset
from torch.utils.data import DataLoader
import numpy as np
from utils import *
import matplotlib.pyplot as plt
from model import SNNModel0HLayers
import pandas as pd
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

num_steps = 25
batch_size = 1

test_dataset = SmartFallDataset('/Users/hemanthsabbella/Documents/SNN_Responsiveness/dataset/smart_fall/time_window_1s_sliding_100ms/test')
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

# Load the first model
snn_model_1 = SNNModel0HLayers(time_steps=num_steps, input_features=6)
snn_model_1.load_state_dict(torch.load("saves/snn_model_1s_25ep_mse_count_loss_quick_encoding_seed0.pt", weights_only=True))
snn_model_1 = snn_model_1.to(device)

# Load the second model
# snn_model_2 = SNNModel0HLayers(time_steps=num_steps, input_features=6)
# snn_model_2.load_state_dict(torch.load("saves/snn_model_1s_40ep_linear_weighted_lc_sampling_seed111.pt", weights_only=True))
# snn_model_2 = snn_model_2.to(device)

# Evaluate model using Quick Spike Encoding
def evaluate_model_qse(model):
    acc_log = []
    with torch.no_grad():
        for _, (inputs, label) in enumerate(test_loader):
            # Encode the inputs
            inputs = quick_spikes_encoding(inputs, 25)
            inputs = torch.cat((inputs, torch.flip(inputs, dims=[2])), dim=1)

            # Check for NaN or Inf in the inputs
            if torch.isnan(inputs).any() or torch.isinf(inputs).any():
                continue

            inputs = inputs.to(device)
            label = label.to(device)
            target = torch.argmax(label, dim=1)

            # Forward pass through the model
            spk = model(inputs)

            # Check for NaN or Inf in the model output
            if torch.isnan(spk).any() or torch.isinf(spk).any():
                continue

            acc_per_step = []
            for i in range(1, len(spk) + 1):
                spikes = spk[0:i]
                acc = SF.accuracy_rate(spikes, target)
                acc_per_step.append(acc.item())

            if acc_per_step:
                acc_log.append(acc_per_step)

    if len(acc_log) == 0:
        print("Error: acc_log is empty for the model.")
        return None

    acc_log = np.asarray(acc_log)
    mean_acc = np.mean(acc_log, axis=0)
    return mean_acc

# Evaluate model using LC Sampling
def evaluate_model_lc(model):
    acc_log = []
    with torch.no_grad():
        for _, (inputs, label) in enumerate(test_loader):
            # Apply LC sampling
            spikes_up_input, spikes_down_input = lc_sampling(inputs)
            inputs = torch.concat((spikes_up_input, spikes_down_input), dim=1)

            # Check for NaN or Inf in the inputs
            if torch.isnan(inputs).any() or torch.isinf(inputs).any():
                continue

            inputs = inputs.to(device)
            label = label.to(device)
            target = torch.argmax(label, dim=1)

            # Forward pass through the model
            spk = model(inputs)

            # Check for NaN or Inf in the model output
            if torch.isnan(spk).any() or torch.isinf(spk).any():
                continue

            acc_per_step = []
            for i in range(1, len(spk) + 1):
                spikes = spk[0:i]
                acc = SF.accuracy_rate(spikes, target)
                acc_per_step.append(acc.item())

            if acc_per_step:
                acc_log.append(acc_per_step)

    if len(acc_log) == 0:
        print("Error: acc_log is empty for the model.")
        return None

    acc_log = np.asarray(acc_log)
    mean_acc = np.mean(acc_log, axis=0)
    return mean_acc

# Evaluate both models
mean_acc_1 = evaluate_model_qse(snn_model_1)
# mean_acc_2 = evaluate_model_lc(snn_model_2)

# Plot the accuracy curves if valid
if mean_acc_1 is not None:
    plt.plot(mean_acc_1, label='Quick_Encoding_MSE_Count_Loss_Seed111')

# if mean_acc_2 is not None:
#     plt.plot(mean_acc_2, label= 'LC_Sampling_Linear_Weighted_Seed111')


file_path = 'files_4/Quick_Encoding_MSE_Count_Loss_Seed0.csv'
timesteps = np.arange(25)
data = {'Timesteps':timesteps,
        'Accuracy':mean_acc_1*100}
df = pd.DataFrame(data)
df.to_csv(file_path, index=False)


plt.xlabel('Time Steps')
plt.ylabel('Accuracy')
plt.title('Accuracy vs Time Steps for Two Models')
plt.legend()
plt.grid()
plt.show()
