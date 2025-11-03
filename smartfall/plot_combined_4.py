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

# Plot all the individual line graphs on the same plot
plt.figure(figsize=(8, 6))

# Assign specific colors for specific files
color_map = {
    "LC Sampling MSE Count Loss": 'green',
    "Quick Encoding MSE Count Loss": 'red',
    "LC Sampling Linear Weighted": 'orange',
    "Quick Encoding Linear Weighted": 'blue'
}

# Map specific files to their legends
legend_mapping = {
    "LC Sampling MSE Count Loss": "LC + MSEC",
    "Quick Encoding MSE Count Loss": "QSE + MSEC",
    "LC Sampling Linear Weighted": "LC + LW-MSEC",
    "Quick Encoding Linear Weighted": "QSE + LW-MSEC"
}

# Specific legend order
legend_order = [
    "LC Sampling MSE Count Loss",
    "Quick Encoding MSE Count Loss",
    "LC Sampling Linear Weighted",
    "Quick Encoding Linear Weighted"
    
]

# Process and plot each file
legend_handles = {}
for file_name, df in data_frames.items():
    # Process the legend name
    raw_legend_name = file_name.replace(".csv", "").replace("_Seed111", "").replace("_", " ")
    legend_name = legend_mapping.get(raw_legend_name, raw_legend_name)

    # Determine the color
    color = color_map.get(raw_legend_name, 'black')  # Map raw legend name to color
    line, = plt.plot(df["Timesteps"], df["Accuracy"], '-o', label=legend_name, linewidth=2, color=color)

    # Store the legend handle with the name
    legend_handles[legend_name] = line

# Sort and display the legend in the specified order
handles = [legend_handles[legend_mapping[name]] for name in legend_order]
labels = [legend_mapping[name] for name in legend_order]

plt.title("Spike Encoding & Loss Function Combinations", fontsize=18)
plt.xlabel("Time Step", fontsize=18)
plt.ylabel("Accuracy (%)", fontsize=18)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.grid(True, linestyle="--", alpha=0.7)
plt.legend(handles, labels, fontsize=14, loc="best")
plt.savefig('plots/smartfall_combinations.png')
plt.show()
