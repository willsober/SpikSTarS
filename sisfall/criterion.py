import snntorch as snn
from snntorch import functional as SF
import torch
import torch.nn as nn
import snntorch.spikegen as spikegen
import math



def linear_weighted_count_loss(spk, label, beta=0.9):
    num_steps, batch_size, _ = spk.shape
    loss_function = SF.mse_count_loss(correct_rate=0.7, incorrect_rate=0.3)
    losses = torch.zeros(num_steps)
    for t in range(1, num_steps+1):
        spikes = spk[0:t]
        loss = loss_function(spikes, label)
        losses[t-1] = loss

    time_steps = torch.arange(num_steps, dtype=torch.float32)
    weights = 1 - beta * (time_steps / (num_steps - 1))
    weights = weights.to(losses.device)
    weighted_loss = torch.sum(weights * losses)
    return weighted_loss


def delayed_linear_weighted_count_loss(spk, label, beta=0.9, delay = 30):
    num_steps, batch_size, _ = spk.shape
    loss_function = SF.mse_count_loss(correct_rate=0.7, incorrect_rate=0.3)
    losses = torch.zeros(num_steps)
    for t in range(1, num_steps+1):
        spikes = spk[0:t]
        loss = loss_function(spikes, label)
        losses[t-1] = loss

    time_steps_1 = torch.ones(delay, dtype=torch.float32)
    time_steps_2 = torch.arange(num_steps-delay, dtype=torch.float32)
    weights_1 = time_steps_1
    weights_2 = 1 - beta * (time_steps_2 / (num_steps - 1))
    weights = torch.cat([weights_1,weights_2])
    weights = weights.to(losses.device)
    numerator = torch.sum(weights * losses)
    denominator = torch.sum(weights)
    weighted_loss = numerator / denominator
    return weighted_loss


