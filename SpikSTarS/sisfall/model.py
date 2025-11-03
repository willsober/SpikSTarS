import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate


class SNNModel0HLayers(torch.nn.Module):

    def __init__(self, time_steps, input_features, beta=0.9):
        super(SNNModel0HLayers, self).__init__()

        self.time_steps = time_steps
        self.input_features = input_features
        self.beta = beta

        # layer 1
        self.fc1 = torch.nn.Linear(in_features=self.input_features, out_features=2)
        self.lif1 = snn.Leaky(beta=self.beta, learn_beta=True)
        self.stsc = STSC(2, dimension=2, time_rf_conv=5, time_rf_at=3, use_gate=True, use_filter=True)


    def forward(self, x):
        """Forward pass for several time steps."""
        #print(x.shape) #torch.Size([4, 6, 25])
        # Initalize membrane potential
        mem1 = self.lif1.init_leaky()


        # Empty lists to record outputs
        spk_recording = []

        for step in range(self.time_steps):
            # print(x.shape)
            input = x[:,:,step]
            # print(input.shape)
            cur1 = self.fc1(input)
            # x3 = cur1.unsqueeze(1)  # 最快
            # # print(x3.shape) #torch.Size([4, 1, 2])
            # x3 = self.stsc(x3)
            # # print(x3.shape) #torch.Size([4, 1, 2])
            #
            # B, F, T = x3.shape
            # cur1 = x3.view(B, -1)
            # print(cur1.shape) #torch.Size([4, 150])
            spk1, mem1 = self.lif1(cur1, mem1)

            spk_recording.append(spk1)

    
        out = torch.stack(spk_recording, dim=0)
        # print(out.shape)
        # # out = torch.stack(mem_recording, dim=0)
        # out = torch.squeeze(out)
        # out = torch.sum(out, dim=0)
        return out

import torch
import torch.nn as nn
from spikingjelly.activation_based import base

class STSC_Attention(nn.Module, base.StepModule):
    def __init__(self, n_channel: int, dimension: int = 4, time_rf: int = 4, reduction: int = 2):

        super().__init__()
        self.step_mode = 'm'  # used in activation_based SpikingJelly
        assert dimension == 4 or dimension == 2, 'dimension must be 4 or 2'

        self.dimension = dimension

        if self.dimension == 4:
            self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.time_padding = (time_rf - 1) // 2
        self.n_channels = n_channel
        r_channel = n_channel // reduction
        self.recv_T = nn.Conv1d(n_channel, r_channel, kernel_size=time_rf, padding=self.time_padding, groups=1,
                                bias=True)
        self.recv_C = nn.Sequential(
            nn.ReLU(),
            nn.Linear(r_channel, n_channel, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x_seq: torch.Tensor):
        assert x_seq.dim() == 3 or x_seq.dim() == 5, ValueError(
            f'expected 3D or 5D input with shape [T, B, N] or [T, B, C, H, W], but got input with shape {x_seq.shape}')
        x_seq_C = x_seq.transpose(0, 1)  # x_seq_C.shape = [B, T, N] or [B, T, C, H, W]
        x_seq_T = x_seq_C.transpose(1, 2)  # x_seq_T.shape = [B, C, N] or [B, C, T, H, W]

        if self.dimension == 2:
            recv_h_T = self.recv_T(x_seq_T)
            recv_h_C = self.recv_C(recv_h_T.transpose(1, 2))
            D_ = 1 - self.sigmoid(recv_h_C)
            D = D_.transpose(0, 1)

        elif self.dimension == 4:
            avgout_C = self.avg_pool(x_seq_C).view(
                [x_seq_C.shape[0], x_seq_C.shape[1], x_seq_C.shape[2]])  # avgout_C.shape = [N, T, C]
            avgout_T = avgout_C.transpose(1, 2)
            recv_h_T = self.recv_T(avgout_T)
            recv_h_C = self.recv_C(recv_h_T.transpose(1, 2))
            D_ = 1 - self.sigmoid(recv_h_C)
            D = D_.transpose(0, 1)

        return D


class STSC_Temporal_Conv(nn.Module, base.StepModule):
    def __init__(self, channels: int, dimension: int = 4, time_rf: int = 2):

        super().__init__()
        self.step_mode = 'm'  # used in activation_based SpikingJelly
        assert dimension == 4 or dimension == 2, 'dimension must be 4 or 2'
        self.dimension = dimension

        time_padding = (time_rf - 1) // 2
        self.time_padding = time_padding

        if dimension == 4:
            kernel_size = (time_rf, 1, 1)
            padding = (time_padding, 0, 0)
            self.conv = nn.Conv3d(channels, channels, kernel_size=kernel_size, padding=padding, groups=channels,
                                  bias=False)
        else:
            kernel_size = time_rf
            self.conv = nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=time_padding, groups=channels,
                                  bias=False)

    def forward(self, x_seq: torch.Tensor):
        assert x_seq.dim() == 3 or x_seq.dim() == 5, ValueError(
            f'expected 3D or 5D input with shape [T, B, N] or [T, B, C, H, W], but got input with shape {x_seq.shape}')

        # x_seq.shape = [T, B, N] or [T, B, C, H, W]

        x_seq = x_seq.transpose(0, 1)  # x_seq.shape = [B, T, N] or [B, T, C, H, W]
        x_seq = x_seq.transpose(1, 2)  # x_seq.shape = [B, N, T] or [B, C, T, H, W]
        x_seq = self.conv(x_seq)
        x_seq = x_seq.transpose(1, 2)  # x_seq.shape = [B, T, N] or [B, T, C, H, W]
        x_seq = x_seq.transpose(0, 1)  # x_seq.shape = [T, B, N] or [T, B, C, H, W]

        return x_seq


class STSC(nn.Module, base.StepModule):
    def __init__(self, in_channel: int, dimension: int = 4, time_rf_conv: int = 3, time_rf_at: int = 3, use_gate=True,
                 use_filter=True, reduction: int = 1):

        super().__init__()
        self.step_mode = 'm'  # used in activation_based SpikingJelly

        assert dimension == 4 or dimension == 2, 'dimension must be 4 or 2'
        self.dimension = dimension

        self.time_rf_conv = time_rf_conv
        self.time_rf_at = time_rf_at

        if use_filter:
            self.temporal_conv = STSC_Temporal_Conv(in_channel, time_rf=time_rf_conv, dimension=dimension)

        if use_gate:
            self.spatio_temporal_attention = STSC_Attention(in_channel, time_rf=time_rf_at, reduction=reduction,
                                                            dimension=dimension)

        self.use_gate = use_gate
        self.use_filter = use_filter

    def forward(self, x_seq: torch.Tensor):
        assert x_seq.dim() == 3 or x_seq.dim() == 5, ValueError(
            f'expected 3D or 5D input with shape [T, B, N] or [T, B, C, H, W], but got input with shape {x_seq.shape}')

        if self.use_filter:
            # Filitering
            x_seq_conv = self.temporal_conv(x_seq)
        else:
            # without filtering
            x_seq_conv = x_seq

        if self.dimension == 2:
            if self.use_gate:
                # Gating
                x_seq_D = self.spatio_temporal_attention(x_seq)
                y_seq = x_seq_conv * x_seq_D
            else:
                # without gating
                y_seq = x_seq_conv
        else:
            if self.use_gate:
                # Gating
                x_seq_D = self.spatio_temporal_attention(x_seq)
                y_seq = x_seq_conv * x_seq_D[:, :, :, None, None]  # broadcast
            else:
                # without gating
                y_seq = x_seq_conv

        return y_seq
