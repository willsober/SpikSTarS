import os
import pandas as pd
import matplotlib.pyplot as plt

# Define the folder path containing CSV files
folder_path = "files/"  # Replace with your actual folder path

# Separate the CSV files into two groups based on their prefixes
quick_encoding_files = [file for file in os.listdir(folder_path) if file.startswith("Quick_Encoding_Linear_Weighted_Seed")]
lc_sampling_files = [file for file in os.listdir(folder_path) if file.startswith("LC_Sampling_MSE_Count_Loss_Seed")]

# Function to read and process a group of files
def process_files(file_list, folder_path):
    data_frames = []
    for file in file_list:
        file_path = os.path.join(folder_path, file)
        df = pd.read_csv(file_path)
        data_frames.append(df)
    combined_data = pd.concat(data_frames)
    combined_data["Accuracy"] = pd.to_numeric(combined_data["Accuracy"], errors="coerce")
    grouped_data = combined_data.groupby("Timesteps")["Accuracy"].agg(['mean', 'std']).reset_index()
    return grouped_data

# Process each group of files
quick_encoding_data = process_files(quick_encoding_files, folder_path)
lc_sampling_data = process_files(lc_sampling_files, folder_path)

print(quick_encoding_data)
print(lc_sampling_data)
# Plot the graphs
plt.figure(figsize=(8, 6))

# Quick Encoding group
plt.plot(quick_encoding_data["Timesteps"], quick_encoding_data["mean"], '-o', label="QSE + LW-MSEC: Mean Accuracy", linewidth=2, color='blue')
plt.fill_between(quick_encoding_data["Timesteps"], 
                 quick_encoding_data["mean"] - quick_encoding_data["std"], 
                 quick_encoding_data["mean"] + quick_encoding_data["std"], 
                 color='blue', alpha=0.2, label="QSE + LW-MSEC: Std Dev")

# LC Sampling group
plt.plot(lc_sampling_data["Timesteps"], lc_sampling_data["mean"], '-o', label="LC + MSEC: Mean Accuracy", linewidth=2, color='green')
plt.fill_between(lc_sampling_data["Timesteps"], 
                 lc_sampling_data["mean"] - lc_sampling_data["std"], 
                 lc_sampling_data["mean"] + lc_sampling_data["std"], 
                 color='green', alpha=0.2, label="LC + MSEC: Std Dev")

plt.title("Mean Accuracy across Timesteps", fontsize=18)
plt.xlabel("Time Step", fontsize=18)
plt.ylabel("Accuracy (%)", fontsize=18)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.grid(True, linestyle="--", alpha=0.7)
plt.legend(fontsize=14)
plt.savefig('plots/sisfall_over_all.png')
plt.show()
