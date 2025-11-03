import os
import pandas as pd
import matplotlib.pyplot as plt

# Define the folder path containing CSV files
folder_path = "files_4/"  # Replace with your actual folder path

# Read all CSV files in the folder and store the data in a dictionary for individual line graphs
data_frames = {}
for file in os.listdir(folder_path):
    if file.endswith(".csv"):
        file_path = os.path.join(folder_path, file)
        df = pd.read_csv(file_path)
        data_frames[file] = df

# Map specific files to their legends
legend_mapping = {
    "Quick_Encoding_Linear_Weighted_Seed0": "QSE + LW-MSEC",
    "LC_Sampling_MSE_Count_Loss_Seed0": "LC + MSEC",
    "Quick_Encoding_MSE_Count_Loss_Seed0": "QSE + MSEC",
    "LC_Sampling_Linear_Weighted_Seed0": "LC + LW-MSEC"
}

# Plot all the individual line graphs on the same plot
plt.figure(figsize=(8, 6))

# Keep track of color assignments for non-specific files
other_colors = iter(['red', 'orange'])

for file_name, df in data_frames.items():
    # Determine the legend based on file name
    legend_name = None
    for key, legend in legend_mapping.items():
        if key in file_name:
            legend_name = legend
            break
    if legend_name is None:  # Fallback for files not in legend mapping
        legend_name = file_name.replace(".csv", "").replace("_Seed0", "").replace("_", " ")
    
    # Set specific colors for designated legends
    if legend_name == "QSE + LW-MSEC":
        plt.plot(df["Timesteps"], df["Accuracy"], '-o', label=legend_name, linewidth=2, color='blue')
    elif legend_name == "LC + MSEC":
        plt.plot(df["Timesteps"], df["Accuracy"], '-o', label=legend_name, linewidth=2, color='green')
    elif legend_name == "QSE + MSEC":
        plt.plot(df["Timesteps"], df["Accuracy"], '-o', label=legend_name, linewidth=2, color='red')
    elif legend_name == "LC + LW-MSEC":
        plt.plot(df["Timesteps"], df["Accuracy"], '-o', label=legend_name, linewidth=2, color='orange')
    else:
        plt.plot(df["Timesteps"], df["Accuracy"], '-o', label=legend_name, linewidth=2, color=next(other_colors))

plt.title("Spike Encoding & Loss Function Combinations", fontsize=18)
plt.xlabel("Time Step", fontsize=18)
plt.ylabel("Accuracy (%)", fontsize=18)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.grid(True, linestyle="--", alpha=0.7)
plt.legend(fontsize=14, loc="best")
plt.savefig('plots/sisfall_combinations.png')
plt.show()
