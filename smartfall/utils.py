import torch
import numpy as np

def calculate_accuracy(outputs, labels):
    # print(f'Outputs: {outputs}')
    # print(f'Labels: {labels}')
    predictions = torch.argmax(outputs, dim=1)
    # print(f'Predictions: {predictions}')
    targets = torch.argmax(labels, dim=1)
    # print(f'Targets: {targets}')
    correct = (predictions == targets).sum().item()
    # print(f'Correct: {correct}')
    accuracy = correct / len(labels)
    # print(accuracy)

    return accuracy

def time_slot_accumulation(spikes_up, sampling_freq, subsampling_freq):
        """
        Performs time slot accumulation on a spike array.

        Args:
            spikes_up: A list or NumPy array of spikes.
            sampling_freq: The sampling frequency of the spikes.
            subsampling_freq: The subsampling frequency for time slot accumulation.

        Returns:
            A list of accumulated spikes.
        """
        batch_size, num_channels, num_samples = spikes_up.shape
        # Calculate the number of samples per time slot
        samples_per_slot = (sampling_freq // subsampling_freq)

        # Initialize the accumulated spikes array
        # accumulated_spikes = [0] * (len(spikes_up) // samples_per_slot)
        accumulated_spikes = torch.zeros([batch_size, num_channels, int(num_samples//(sampling_freq/subsampling_freq))])

        # Iterate through the spikes array
        # for i in range(0, len(spikes_up), samples_per_slot):
        #     # Check if there is a spike in the current time slot
        #     if any(spikes_up[i:i+samples_per_slot]):
        #         accumulated_spikes[i // samples_per_slot] = 1

        for batch in range(batch_size):
            for channel in range(num_channels):
                for i in range(0, num_samples, samples_per_slot):

                    # Check if there is a spike in the current time slot
                    if any(spikes_up[batch, channel,i:i+samples_per_slot]):
                        accumulated_spikes[batch, channel, i // samples_per_slot] = 1.0


        return accumulated_spikes

def lc_sampling(sensor_values, n_levels=20):
    """
    Performs level crossing sampling on normalized sensor values, returning separate spikes for increases and decreases.
    
    Args:
        sensor_values: A NumPy array of shape [batch_size, number_of_channels, sequence_length].
        n_levels: The number of levels to use for sampling.
        
    Returns:
        A tuple of two NumPy arrays:
        - The first array contains spikes when the sensor value increases at a level crossing (shape: [batch_size, number_of_channels, sequence_length]).
        - The second array contains spikes when the sensor value decreases at a level crossing (shape: [batch_size, number_of_channels, sequence_length]).
    """
    batch_size, num_channels, seq_len = sensor_values.shape
    # print(f"Input shape: {sensor_values.shape}")

    # Calculate level spacing
    level_spacing = 1.0 / n_levels

    # Create a list of levels
    levels = np.linspace(level_spacing, 1.0, n_levels)

    # Initialize spikes arrays
    spikes_up = torch.zeros((batch_size, num_channels, seq_len), dtype=torch.float32)
    spikes_down = torch.zeros((batch_size, num_channels, seq_len), dtype=torch.float32)

    # Iterate through each batch and channel
    for batch in range(batch_size):
        for channel in range(num_channels):
            # Iterate through the sensor values
            for i in range(1, seq_len):
                # Check if the previous value was below a level and the current value is above (for spikes_up)
                for level in levels:
                    if sensor_values[batch, channel, i - 1] < level and sensor_values[batch, channel, i] >= level:
                        spikes_up[batch, channel, i] = 1.0
                        break

                # Check if the previous value was above a level and the current value is below (for spikes_down)
                for level in levels:
                    if sensor_values[batch, channel, i - 1] > level and sensor_values[batch, channel, i] <= level:
                        spikes_down[batch, channel, i] = 1.0
                        break
    # spikes_up = torch.tensor(spikes_up, dtype=torch.float)
    # spikes_down = torch.tensor(spikes_down, dtype=torch.float)
    return spikes_up, spikes_down

def quick_spikes_encoding(input_sequence, num_thresholds=25):
    """
    Generates a spike train based on non-linear threshold levels, where a spike is set 
    each time the input crosses a threshold from one level to another.
    
    Parameters:
    - input_sequence: array-like, normalized time series input between 0 and 1.
    - num_thresholds: int, total number of threshold levels (default is 25).
    
    Returns:
    - spike_train: binary array with spikes generated based on threshold crossings.
    - thresholds: array of threshold values used for reference in plotting.
    """
    # Define non-linear thresholds closer to 1
    # thresholds = torch.tensor(1 - (np.linspace(0.05, 0.8, num_thresholds) ** 2)[::-1], dtype=torch.float32)
    thresholds = torch.tensor(np.linspace(0.3, 0.85, num_thresholds), dtype=torch.float32)
    # print(thresholds)
    # Initialize spike train with zeros
    batch_size, num_channels, seq_len = input_sequence.shape

    spike_train = torch.zeros((batch_size, num_channels, num_thresholds), dtype=torch.float32)

    for batch in range(batch_size):
        for channel in range(num_channels):
            for i, threshold in enumerate(reversed(thresholds)):
                # Using torch.any for PyTorch tensors
                if torch.any(input_sequence[batch, channel, :] >= threshold):
                    spike_train[batch, channel, i] = 1
                    # print(spike_train[batch, channel, i])
                else:
                    spike_train[batch, channel, i] = 0  # Keep it zero if threshold isn't met


    return spike_train
