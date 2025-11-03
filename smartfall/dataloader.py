import torch
from torch.utils.data import Dataset, DataLoader
import os
import pandas as pd
import glob
from snntorch import spikegen
from utils import *  # Import all utilities from your utils module
import matplotlib.pyplot as plt
import numpy as np


class SmartFallDataset(Dataset):
    def __init__(self, dataset_path):
        # Get all CSV files from the specified path
        self.data_files = sorted(glob.glob(dataset_path + '/*.csv'))

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx):
        # Read the CSV file
        data_file = self.data_files[idx]
        df = pd.read_csv(data_file)

        # Strip any leading or trailing spaces from the column names
        df.columns = df.columns.str.strip()

        # Extract the accelerometer data
        data = torch.tensor(df[['ms_accelerometer_x', 'ms_accelerometer_y', 'ms_accelerometer_z']].values[:25], dtype=torch.float32)
        label_index = df['outcome'].values[0]

        # Permute the data to have shape (3, number_of_samples)
        data = torch.permute(data, (1, 0))

        # One-hot encode the label
        label = torch.zeros(2, dtype=torch.float32)
        label[label_index] = 1.0

        return data, label




if __name__ == "__main__":

    # Paths to the dataset
    data_path = '/Users/hemanthsabbella/Documents/SNN_Responsiveness/dataset/smart_fall/time_window_1s_sliding_no/test'
    dataset = SmartFallDataset(data_path)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    # Choose the appropriate device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

    # Iterate through the DataLoader
    for _, (inputs, label) in enumerate(dataloader):
        inputs = inputs.to(device)
        label = label.to(device)

        # Generate spike encodings
        spikes_up_input, spikes_down_input = lc_sampling(inputs)  # Replace with your LC sampling function
        spikes_quick = quick_spikes_encoding(inputs, num_thresholds=25)  # Replace with your quick spike encoding function
        # spikes_rate = spikegen.rate(inputs, num_steps=1)
        print(spikes_quick.shape, spikes_up_input.shape, spikes_down_input.shape)
        # spikes_rate = spikes_rate.reshape(1, 3, 30)

        # Prepare time steps and target
        time_steps = torch.arange(25)
        target = torch.argmax(label, dim=1)
        print(target)
        # Adjust spike data for plotting
        spikes_quick = spikes_quick[0, 0] * time_steps
        spikes_up = spikes_up_input[0, 0] * time_steps
        spikes_down = spikes_down_input[0, 0] * time_steps
        # spikes_rate = spikes_rate[0, 0] * time_steps

        # Plotting with adjusted font size and line width
        plt.figure(figsize=(10, 12))  # Increase figure size

        # Set common parameters for font size and line width
        font_size = 22
        font_size_2 = 18
        line_width = 5
        tick_font_size = 22  # Font size for tick labels

        # Plot Accelerometer Data
        plt.subplot(3, 1, 1)
        plt.plot(time_steps, inputs[0, 0].cpu().numpy(), linewidth=line_width)
        plt.xlabel('Time Steps', fontsize=font_size_2)
        plt.xticks(fontsize=tick_font_size)
        plt.yticks(fontsize=tick_font_size)
        plt.xlim([0.1, 25.0])
        plt.ylim([0.0, 1.0])
        plt.title('Accelerometer Data', fontsize=font_size)
        plt.grid()

        # Plot LC Sampling Spikes
        plt.subplot(3, 1, 2)
        plt.eventplot(spikes_up.cpu().numpy(), color='red', linewidths=line_width)
        plt.eventplot(spikes_down.cpu().numpy(), color='blue', linewidths=line_width)
        plt.xlabel('Time Steps', fontsize=font_size_2)
        plt.xticks(fontsize=tick_font_size)
        plt.yticks(fontsize=tick_font_size)
        plt.xlim([0.1, 25.0])
        plt.title('LC Sampling', fontsize=font_size)
        plt.grid()

        # Plot Rate Encoding Spikes
        # plt.subplot(4, 1, 3)
        # plt.eventplot(spikes_rate.cpu().numpy(), colors='purple', linewidths=line_width)
        # plt.xlabel('Time Steps', fontsize=font_size_2)
        # plt.xticks(fontsize=tick_font_size)
        # plt.yticks(fontsize=tick_font_size)
        # plt.xlim([0.1, 50.0])
        # plt.title('Rate Encoding', fontsize=font_size)
        # plt.grid()

        # Plot Quick Spike Encoding
        plt.subplot(3, 1, 3)
        plt.eventplot(spikes_quick.cpu().numpy(), colors='green', linewidths=line_width)
        plt.xlabel('Time Steps', fontsize=font_size_2)
        plt.xticks(fontsize=tick_font_size)
        plt.yticks(fontsize=tick_font_size)
        plt.xlim([0.1, 25.0])
        plt.title('Quick Spike Encoding', fontsize=font_size)
        plt.grid()

        # Use plt.tight_layout() with padding
        plt.tight_layout(pad=6.0)  # Increase the padding between subplots

        plt.show()
