import torch
from torch.utils.data import Dataset, DataLoader
import os
import pandas as pd
import glob
from snntorch import spikegen
from utils import *  # Import all utilities from your utils module
import matplotlib.pyplot as plt
import numpy as np

class SisFallDataset(Dataset):
    def __init__(self, dataset_path):
        self.data_files = sorted(glob.glob(dataset_path + '/*.csv'))

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx):
        data_file = self.data_files[idx]
        df = pd.read_csv(data_file)

        data = torch.tensor(df[['a1_x', 'a1_y', 'a1_z']].values, dtype=torch.float32)
        label_index = df['label'].values[0]

        data = torch.permute(data, (1, 0))
        label = torch.zeros(2, dtype=torch.float32)
        label[label_index] = 1.0

        # Down Sampling
        original_sampling_rate = 200
        new_sampling_rate = 100
        num_samples = int(original_sampling_rate / new_sampling_rate)
        data = data[:, ::num_samples]

        return data, label


if __name__ == "__main__":
    data_path = '/Users/hemanthsabbella/Documents/SNN_Responsiveness/dataset/sis_fall/time_window_500ms_sliding_50ms/test'
    dataset = SisFallDataset(data_path)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

    for _, (inputs, label) in enumerate(dataloader):
        spikes_up_input, spikes_down_input = lc_sampling(inputs)
        spikes_quick = quick_spikes_encoding(inputs, num_thresholds=50)
        spikes_rate = spikegen.rate(inputs, num_steps=1)
        spikes_rate = spikes_rate.reshape(1, 3, 50)

        # Prepare time steps and target
        time_steps = torch.arange(50)
        target = torch.argmax(label, dim=1)

        # Adjust spike data for plotting
        spikes_quick = spikes_quick[0, 0] * time_steps
        spikes_up = spikes_up_input[0, 0] * time_steps
        spikes_down = spikes_down_input[0, 0] * time_steps
        spikes_rate = spikes_rate[0, 0] * time_steps

        # Plotting with adjusted font size and line width
        plt.figure(figsize=(10, 12))  # Increase figure size

        # Set common parameters for font size and line width
        font_size = 22
        font_size_2 = 18
        line_width = 5
        tick_font_size = 22  # Font size for tick labels

        plt.subplot(4, 1, 1)
        plt.plot(time_steps, inputs[0, 0].numpy(), linewidth=line_width)
        plt.xlabel('Time Steps', fontsize=font_size_2)
        plt.xticks(fontsize=tick_font_size)  # Set x-axis tick font size
        plt.yticks(fontsize=tick_font_size)  # Set y-axis tick font size
        plt.xlim([0.1, 50.0])
        plt.ylim([0.0, 1.0])
        plt.title('Accelerometer Data', fontsize=font_size)
        plt.grid()

        plt.subplot(4, 1, 2)
        plt.eventplot(spikes_up.numpy(), color='red', linewidths=line_width)
        plt.eventplot(spikes_down.numpy(), color='blue', linewidths=line_width)
        plt.xlabel('Time Steps', fontsize=font_size_2)
        plt.xticks(fontsize=tick_font_size)  # Set x-axis tick font size
        plt.yticks(fontsize=tick_font_size)  # Set y-axis tick font size
        plt.xlim([0.1, 50.0])
        plt.title('LC Sampling', fontsize=font_size)
        plt.grid()

        plt.subplot(4, 1, 3)
        plt.eventplot(spikes_rate.numpy(), colors='purple', linewidths=line_width)
        plt.xlabel('Time Steps', fontsize=font_size_2)
        plt.xticks(fontsize=tick_font_size)  # Set x-axis tick font size
        plt.yticks(fontsize=tick_font_size)  # Set y-axis tick font size
        plt.xlim([0.1, 50.0])
        plt.title('Rate Encoding', fontsize=font_size)
        plt.grid()

        plt.subplot(4, 1, 4)
        plt.eventplot(spikes_quick.numpy(), colors='green', linewidths=line_width)
        plt.xlabel('Time Steps', fontsize=font_size_2)
        plt.xticks(fontsize=tick_font_size)  # Set x-axis tick font size
        plt.yticks(fontsize=tick_font_size)  # Set y-axis tick font size
        plt.xlim([0.1, 50.0])
        plt.title('Quick Spike Encoding', fontsize=font_size)
        plt.grid()

        # Use plt.tight_layout() with padding
        plt.tight_layout(pad=6.0)  # Increase the padding between subplots

        plt.show()
