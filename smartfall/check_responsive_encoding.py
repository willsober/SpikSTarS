import numpy as np
import matplotlib.pylab as plt


def generate_spike_train(input_sequence, num_thresholds):
    """
    Generates a spike train based on non-linear threshold levels.
    
    Parameters:
    - input_sequence: array-like, normalized time series input between 0 and 1.
    - num_thresholds: int, number of non-linear threshold levels.
    
    Returns:
    - spike_train: binary array with spikes generated based on threshold levels.
    """
    # Define non-linear thresholds closer near 1 and wider near 0
    thresholds = np.linspace(0, 1, num_thresholds) ** 2  # non-linear scale
    print(thresholds)
    # Initialize spike train with zeros
    spike_train = np.zeros_like(input_sequence)
    
    # Iterate over each threshold from highest to lowest
    for i, threshold in enumerate(reversed(thresholds)):
        # Find indices where input values are greater than or equal to the current threshold
        spike_indices = np.where(input_sequence >= threshold)[0]
        
        # Replace the corresponding zero in the spike train with one
        for idx in spike_indices:
            if spike_train[idx] == 0:  # only replace the first zero found
                spike_train[idx] = 1
                break  # move to the next threshold after setting the first spike
    
    return spike_train

# Generate two different input sequences: one with high variations and one relatively flat
high_variation_input = np.random.rand(50)  # high variations
flat_input = np.full(50, 0.5) + 0.05 * np.random.randn(50)  # relatively flat around 0.5 with small noise

# Generate spike trains for both inputs
high_variation_spike_train = generate_spike_train(high_variation_input, num_thresholds=50)
flat_spike_train = generate_spike_train(flat_input, num_thresholds=50)

# Plot both input sequences and their respective spike trains
plt.figure(figsize=(12, 10))

# Plot high variation input sequence
plt.subplot(4, 1, 1)
plt.plot(high_variation_input, label='High Variation Input')
plt.title('High Variation Input Sequence')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()

# Plot spike train for high variation input
plt.subplot(4, 1, 2)
plt.plot(high_variation_spike_train, label='Spike Train for High Variation Input', color='orange')
plt.title('Spike Train for High Variation Input')
plt.xlabel('Time')
plt.ylabel('Spike')
plt.legend()

# Plot relatively flat input sequence
plt.subplot(4, 1, 3)
plt.plot(flat_input, label='Flat Input', color='green')
plt.title('Relatively Flat Input Sequence')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()

# Plot spike train for relatively flat input
plt.subplot(4, 1, 4)
plt.plot(flat_spike_train, label='Spike Train for Flat Input', color='red')
plt.title('Spike Train for Flat Input')
plt.xlabel('Time')
plt.ylabel('Spike')
plt.legend()

plt.tight_layout()
plt.show()

