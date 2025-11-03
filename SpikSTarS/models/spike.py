import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.backbones import FCN, DeepConvLSTM
import torch.nn.functional as F
from torch.nn import init
# from SE_adLIF import SEAdLIF
import argparse


class AvgMeter:

    def __init__(self):
        self.value = 0
        self.number = 0

    def add(self, v, n):
        self.value += v
        self.number += n

    def avg(self):
        return self.value / self.number


class ZIF(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, gamma):
        out = (input > 0).float()
        L = torch.tensor([gamma])
        ctx.save_for_backward(input, out, L)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        (input, out, others) = ctx.saved_tensors
        gamma = others[0].item()
        grad_input = grad_output.clone()
        tmp = (1 / gamma) * (1 / gamma) * ((gamma - input.abs()).clamp(min=0))
        # tmp = torch.ones_like(input)
        # tmp = torch.where(input.abs() < 0.5, 1., 0.)
        grad_input = grad_input * tmp
        return grad_input, None


class DSPIKE(nn.Module):
    def __init__(self, region=1.0):
        super(DSPIKE, self).__init__()
        self.region = region

    def forward(self, x, temp):
        out_bp = torch.clamp(x, -self.region, self.region)
        out_bp = (torch.tanh(temp * out_bp)) / \
                 (2 * np.tanh(self.region * temp)) + 0.5
        out_s = (x >= 0).float()
        return (out_s.float() - out_bp).detach() + out_bp


class LIFSpike(nn.Module):
    def __init__(self, thresh=0.5, tau=0.5, gamma=1.0, dspike=False, soft_reset=True,beta=0.9,a=0.5,b=0.5):
        """
        Implementing the LIF neurons.
        @param thresh: firing threshold;
        @param tau: membrane potential decay factor;
        @param gamma: hyper-parameter for controlling the sharpness in surrogate gradient;
        @param dspike: whether using rectangular gradient of dspike gradient;
        @param soft_reset: whether using soft-reset or hard-reset.
        """
        super(LIFSpike, self).__init__()
        if not dspike:
            self.act = ZIF.apply
        else:
            # using the surrogate gradient function from Dspike: 
            # https://proceedings.neurips.cc/paper/2021/file/c4ca4238a0b923820dcc509a6f75849b-Paper.pdf
            self.act = DSPIKE(region=1.0)
        self.thresh = thresh
        self.tau = tau
        self.gamma = gamma
        self.soft_reset = soft_reset
        self.soft_reset = soft_reset
        self.beta = beta  # 控制自适应电流
        self.a = a  # For adaptive current update
        self.b = b  # For adaptive current update



    def forward(self, x):
        mem = 0
        w=0
        # Initialize membrane potential and adaptive current
        #mem = torch.zeros(x.shape[0], x.shape[1], device=x.device)  # Membrane potential
        #w = torch.zeros_like(mem)  # Adaptive current
        spike_out = []

       # spike_out = self.neu(x)
        T = x.shape[2]
        #print(x.shape)
        #获取时间步数T
        for t in range(T):
            #遍历每个时间步，模拟神经元在时间上的动态变化
            mem = mem * self.tau + x[:, :, t]-w
            # 更新膜电位，先按照衰减因子tau衰减，然后加上当前时间步的输入
            spike = self.act(mem - self.thresh, self.gamma) #使用self.act 激活函数

            # 自适应电流在每个时间步长更新，基于当前膜电位和脉冲输出
            w = self.beta * w + (1.0 - self.beta) * (self.a * mem + self.b * spike)

            #计算脉冲输出，然后通过激活函数self.act 处理，将膜电位减去阈值
            mem = mem - spike * self.thresh if self.soft_reset else (1 - spike) * mem
            #根据重置方式更新膜电位
            spike_out.append(spike)

        return torch.stack(spike_out, dim=2)


class SFCN(FCN):

    def __init__(self, n_channels, n_classes, out_channels=128, backbone=True, **kwargs):
        super(SFCN, self).__init__(n_channels, n_classes, out_channels, backbone)
        self.conv_block1 = nn.Sequential(nn.Conv1d(n_channels, 32, kernel_size=8, stride=1, bias=False, padding=4),
                                         nn.BatchNorm1d(32),
                                         LIFSpike(**kwargs),
                                         nn.AvgPool1d(kernel_size=2, stride=2, padding=1),
                                         nn.Dropout(0.35))
        self.conv_block2 = nn.Sequential(nn.Conv1d(32, 64, kernel_size=8, stride=1, bias=False, padding=4),
                                         nn.BatchNorm1d(64),
                                         LIFSpike(**kwargs),
                                         nn.AvgPool1d(kernel_size=2, stride=2, padding=1))
        self.conv_block3 = nn.Sequential(nn.Conv1d(64, out_channels, kernel_size=8, stride=1, bias=False, padding=4),
                                         nn.BatchNorm1d(out_channels),
                                         LIFSpike(**kwargs),
                                         nn.AvgPool1d(kernel_size=2, stride=2, padding=1))
        #self.fc = nn.Linear(out_channels, n_classes)




class SDCL(DeepConvLSTM):

    def __init__(self, n_channels, n_classes, conv_kernels=64, kernel_size=5, LSTM_units=128, backbone=True, **snn_p):
        super(SDCL, self).__init__(n_channels, n_classes, conv_kernels, kernel_size, LSTM_units, backbone)
        self.act1 = LIFSpike(**snn_p)
        self.act2 = LIFSpike(**snn_p)
        self.act3 = LIFSpike(**snn_p)
        self.act4 = LIFSpike(**snn_p)

        self.bn1 = nn.BatchNorm2d(conv_kernels)
        self.bn2 = nn.BatchNorm2d(conv_kernels)
        self.bn3 = nn.BatchNorm2d(conv_kernels)
        self.bn4 = nn.BatchNorm2d(conv_kernels)

        self.dropout = nn.Dropout(0.0)

        #self.aft = AFT_FULL(d_model=3,n=151) #第一个模型
        #self.sea=SEAttention (channel= 3)
        self.star = STAR(d_series=192, d_core=64) #A+B
        #self.cross = CrossDeformAttn(seq_len=151, d_model=3, n_heads=8, dropout=1, droprate=1, n_days=1) #第三个

        # 第四个 seq_len:历史时间步长;  pred_len:预测时间步长;  top_k:选择K个频率;  d_model:通道;  d_ff:inception Conv中的通道;  num_kernels: inception中的卷积层个数

        #self.timesblock = TimesBlock(seq_len=151, pred_len=151, top_k=5, d_model=3, d_ff=3, num_kernels=3)
        #self.effi = EfficientAdditiveAttnetion(in_dims=576, token_dim=576)
        self.stsc = STSC(192, dimension=2, time_rf_conv=5, time_rf_at=3, use_gate=False, use_filter=False)
        #self.cbma = CBAM(64)

            # output = block(input)  # 通过CBAM模块处理输入特征图


        # self.agent = AgentAttention(dim=3, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.,
        #                        agent_num=49, window=14)
        # H = W = int(151 ** 0.5)  # H*W=L; H和W是基于2D特征图的高和宽
        # self.mll = MLLABlock(dim=3, input_resolution=(H, W), num_heads=8) # input_resolution=(H,W);  H*W=L; H和W是基于2D特征图的高和宽

    def forward(self, x):
        # B,N,C=x.size()
        # print(x.shape)
        # H,W=64,64
        # total_elements = N * C
        # H, W = 64, 64
        # new_N = total_elements // (H * W)
        # x_reshape = x.view(B, new_N, H, W)



        # print(x_reshape.shape)
        #x= self.sea(x_reshape)


        #x=self.aft(x) #第一个模块
        #x = self.star(x)  # (B,D,L)-->(B,D,L)
        #x = self.cross(x)  # (B,L,D)-->(B,L,D)
        #x = self.timesblock(x)  # (B,T,N)--> (B,T,N)
        #

        #x = self.agent(x)
        #x = self.mll(x)
        # print(x.shape)
        self.lstm.flatten_parameters()#涨点

        #print(x.shape)

        x = x.unsqueeze(1)
        #print(x.shape)
        x = self.act1(self.bn1(self.conv1(x)))
        x = self.act2(self.bn2(self.conv2(x)))
        x = self.act3(self.bn3(self.conv3(x)))
        x = self.act4(self.bn4(self.conv4(x)))


        #print(x.shape)

        x = x.permute(2, 0, 3, 1)
        # print(x.shape) #torch.Size([151, 64, 3, 64])
        # print(x.shape)

        # x = self.stsc(x)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        # print(x.shape) torch.Size([151, 64, 192])
        # x = self.stsc(x)
        # print(x.shape)
        #x = self.se(x)
        #x = self.effi(x)
        x = self.stsc(x)
        # print(x.shape)
        x = self.star(x)
        #print(x.shape)
        x = self.dropout(x)
       # print(x.shape) #torch.Size([128, 64, 576])


        x, h = self.lstm(x)
        x = x[-1, :, :]

        if self.backbone:
            return None, x
        else:
            out = self.classifier(x)
            return out, x


#
# class SDCL(DeepConvLSTM):
#
#     def __init__(self, args,n_channels, n_classes, conv_kernels=64, kernel_size=5, LSTM_units=128, backbone=True, **snn_p):
#         super(SDCL, self).__init__(n_channels, n_classes, conv_kernels, kernel_size, LSTM_units, backbone)
#
#         self.act1 = SEAdLIF(args)
#         self.act2 = SEAdLIF(args)
#         self.act3 = SEAdLIF(args)
#         self.act4 = SEAdLIF(args)
#
#         # self.act1 = SEAdLIF(args, **snn_p)
#         # self.act2 = SEAdLIF(args, **snn_p)
#         # self.act3 = SEAdLIF(args, **snn_p)
#         # self.act4 = SEAdLIF(args, **snn_p)
#         self.bn1 = nn.BatchNorm2d(conv_kernels)
#         self.bn2 = nn.BatchNorm2d(conv_kernels)
#         self.bn3 = nn.BatchNorm2d(conv_kernels)
#         self.bn4 = nn.BatchNorm2d(conv_kernels)
#
#         self.dropout = nn.Dropout(0.0)
#
#
#         self.stsc = STSC(576, dimension=2, time_rf_conv=5, time_rf_at=3, use_gate=True, use_filter=True)
#
#     def forward(self, x):
#
#         self.lstm.flatten_parameters()
#
#         #print(x.shape)
#
#         x = x.unsqueeze(1)
#         #print(x.shape)
#         x = self.act1.layer_forward(self.bn1(self.conv1(x)))
#
#         #x = self.cbma(x)
#         #torch.Size([64, 64, 128, 9])
#         x = self.act2.layer_forward(self.bn2(self.conv2(x)))
#         x = self.act3.layer_forward(self.bn3(self.conv3(x)))
#         x = self.act4.layer_forward(self.bn4(self.conv4(x)))
#
#
#
#
#
#
#         x = x.permute(2, 0, 3, 1)
#         x = x.reshape(x.shape[0], x.shape[1], -1)
#
#         x = self.dropout(x)
#
#
#
#         x, h = self.lstm(x)
#         x = x[-1, :, :]
#
#         if self.backbone:
#             return None, x
#         else:
#             out = self.classifier(x)
#             return out, x
#
#


# 定义XNorm函数，对输入x进行规范化
def XNorm(x, gamma):
    norm_tensor = torch.norm(x, 2, -1, True)
    return x * gamma / norm_tensor

# UFOAttention类继承自nn.Module
class UFOAttention(nn.Module):
    '''
    实现一个改进的自注意力机制，具有线性复杂度。
    '''

    # 初始化函数
    def __init__(self, d_model, d_k, d_v, h, dropout=.1):
        '''
        :param d_model: 模型的维度
        :param d_k: 查询和键的维度
        :param d_v: 值的维度
        :param h: 注意力头数
        '''
        super(UFOAttention, self).__init__()
        # 初始化四个线性层：为查询、键、值和输出转换使用
        self.fc_q = nn.Linear(d_model, h * d_k)
        self.fc_k = nn.Linear(d_model, h * d_k)
        self.fc_v = nn.Linear(d_model, h * d_v)
        self.fc_o = nn.Linear(h * d_v, d_model)
        self.dropout = nn.Dropout(dropout)
        # gamma参数用于规范化
        self.gamma = nn.Parameter(torch.randn((1, h, 1, 1)))

        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.h = h

        self.init_weights()

    # 权重初始化
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    # 前向传播
    def forward(self, queries, keys, values):
        b_s, nq = queries.shape[:2]
        nk = keys.shape[1]

        # 通过线性层将查询、键、值映射到新的空间
        q = self.fc_q(queries).view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)
        k = self.fc_k(keys).view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)
        v = self.fc_v(values).view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)

        # 计算键和值的乘积，然后对结果进行规范化
        kv = torch.matmul(k, v)  # bs,h,c,c
        kv_norm = XNorm(kv, self.gamma)  # bs,h,c,c
        q_norm = XNorm(q, self.gamma)  # bs,h,n,c
        out = torch.matmul(q_norm, kv_norm).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)
        out = self.fc_o(out)  # (b_s, nq, d_model)

        return out

#我自己的模块

#现有的 STAR 模块基础上，加入轻量级个体偏置校正模块（Personalized Bias Correction Layer, PBCL），以适应用户个体差异，提升模型泛化性。
class PBCL(nn.Module):
    def __init__(self, d_series, reduction=4):
        super(PBCL, self).__init__()
        # d_series是通道数
        self.fc1 = nn.Linear(d_series, d_series // reduction)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(d_series // reduction, d_series)
        self.sigmoid = nn.Sigmoid()  # 用sigmoid让偏置在0~1范围内，可以调节强度

    def forward(self, x):
        # x: (B, D, L)
        # 先对序列长度维度做全局平均池化，得到 (B, D)
        x_pool = x.mean(dim=2)  # (B, D)

        # 生成偏置权重 (B, D)
        bias = self.fc1(x_pool)
        bias = self.relu(bias)
        bias = self.fc2(bias)
        bias = self.sigmoid(bias)  # (B, D)

        # 扩展偏置维度到 (B, D, 1)，方便广播
        bias = bias.unsqueeze(2)

        # 偏置加到输入特征上，做动态校正
        out = x + bias

        return out
#第三个模块
class STAR(nn.Module):
    def __init__(self, d_series, d_core):
        super(STAR, self).__init__()
        """
        STar Aggregate-Redistribute Module
        """

        self.gen1 = nn.Linear(d_series, d_series)
        self.gen2 = nn.Linear(d_series, d_core)
        self.gen3 = nn.Linear(d_series + d_core, d_series)
        self.gen4 = nn.Linear(d_series, d_series)
        # self.aft = AFT_FULL(d_model=3, n=151)
    def forward(self, input, *args, **kwargs):
        batch_size, channels, d_series = input.shape  # (B,D,L)
        #print(input.shape)
        # input = self.aft(input)
        # set FFN
        combined_mean = F.gelu(self.gen1(input)) # (B,D,L)-->(B,D,L)
        combined_mean = self.gen2(combined_mean) # (B,D,L)-->(B,D,L_core)

        # stochastic pooling
        if self.training:
            ratio = F.softmax(combined_mean, dim=1) # 在通道方向上执行softmax,为随机池化生成一个概率权重: (B,D,L_core)-->(B,D,L_core)
            ratio = ratio.permute(0, 2, 1) # (B,D,L_core)--permute->(B,L_core,D)
            ratio = ratio.reshape(-1, channels) # 转换为2维, 便于进行采样: (B,L_core,D)--reshape-->(B*L_core,D)
            indices = torch.multinomial(ratio, 1) # 从多项分布ratio的每一行中抽取一个样本,返回值是采样得到的类别的索引: (B*L_core,1); 输入如果是一维张量,它表示每个类别的概率;如果是二维张量,每行表示一个概率分布
            indices = indices.view(batch_size, -1, 1).permute(0, 2, 1) # (B*L_core,1)--view--> (B,L_core,1)--permute-->(B,1,L_core)
            combined_mean = torch.gather(combined_mean, 1, indices) # 根据索引indices在D方向上选择对应的通道元素(理解为:选择重要的通道信息): (B,D,L_core)--gather-->(B,1,L_core)    # gather函数不了解的看这个:https://zhuanlan.zhihu.com/p/661293803
            combined_mean = combined_mean.repeat(1, channels, 1) # 复制D份,将随机选择的core表示应用到所有通道上: (B,1,L_core)--repeat-->(B,D,L_core)
        else:
            weight = F.softmax(combined_mean, dim=1) # 处于非训练模式时, 首先通过softmax生成一个权重分布:(B,D,L_core)-->(B,D,L_core)
            combined_mean = torch.sum(combined_mean * weight, dim=1, keepdim=True).repeat(1, channels, 1) # 直接在D方向上进行加权求和, 然后复制D份: (B,D,L_core)--sum-->(B,1,L_core)--repeat-->(B,D,L_core)

        # mlp fusion
        combined_mean_cat = torch.cat([input, combined_mean], -1) # (B,D,L)--cat--(B,D,L_core)==(B,D,L+L_core)
        combined_mean_cat = F.gelu(self.gen3(combined_mean_cat)) # (B,D,L+L_core)-->(B,D,L)
        combined_mean_cat = self.gen4(combined_mean_cat) # (B,D,L)-->(B,D,L)
        output = combined_mean_cat

        return output





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

