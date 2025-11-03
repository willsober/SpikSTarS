from typing import Optional, Sequence, Tuple
import torch
from torch.nn import Module
from torch import Tensor
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from models.helpers import  spike_grad_injection_function, generic_scan, generic_scan_with_states
from module.tau_trainers import TauTrainer, get_tau_trainer_class
from omegaconf import DictConfig

class EFAdLIF(Module):
    __constants__ = ["in_features", "out_features"]
    in_features: int
    out_features: int
    a: Tensor
    b: Tensor 
    weight: Tensor

    def __init__(
        self,
        cfg: DictConfig,
        device=None,
        dtype=None,
        **kwargs,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__(**kwargs)
        self.in_features = cfg.input_size
        self.out_features = cfg.n_neurons
        self.dt =  cfg.get('dt', 1.0)
        thr = cfg.get('thr', 1.0)
        self.unroll = cfg.get('unroll', 10)
        if isinstance(thr, Sequence):
            thr = torch.FloatTensor(self.out_features, device=device).uniform_(thr[0], thr[1])
        else:
            thr = Tensor([thr,])
        if cfg.get('train_thr', False):
            self.thr = Parameter(thr)
        else:
            self.register_buffer('thr', thr)
        self.alpha = cfg.get('alpha', 5.0)
        self.c = cfg.get('c', 0.4)
        self.tau_u_range = cfg.tau_u_range
        self.train_tau_u_method = cfg.get("train_tau", 'interpolation')
        self.tau_w_range = cfg.tau_w_range
        self.train_tau_w_method = cfg.get("train_tau", 'interpolation')        
        self.use_recurrent = cfg.get('use_recurrent', True)
        
        self.ff_gain = cfg.get('ff_gain', 1.0)
        self.a_range =  cfg.get('a_range', [0.0, 1.0])
        self.b_range = cfg.get('b_range',[0.0, 2.0])
        self.num_out_neuron = cfg.get('num_out_neuron', self.out_features)
        self.use_u_rest = cfg.get('use_u_rest', False)
        self.train_u0 = cfg.get('train_u0', False)

        self.q = cfg.q
        
        self.tau_u_trainer: TauTrainer = get_tau_trainer_class(self.train_tau_u_method)(
                self.out_features,
                self.dt, 
                self.tau_u_range[0], 
                self.tau_u_range[1],
                **factory_kwargs)
        
        self.tau_w_trainer: TauTrainer = get_tau_trainer_class(self.train_tau_w_method)(
                self.out_features,
                self.dt, 
                self.tau_w_range[0], 
                self.tau_w_range[1],
                **factory_kwargs)
        
        
        self.weight = Parameter(
            torch.empty((self.out_features, self.in_features), **factory_kwargs)
        )
        self.bias = Parameter(torch.empty(self.out_features, **factory_kwargs))
        
        if self.use_recurrent:
            self.recurrent = Parameter(
                torch.empty((self.out_features, self.out_features), **factory_kwargs)
            )
        else:
            # registering an empty size tensor is required for the static analyser when using jit.script
            self.register_buffer("recurrent", torch.empty(size=()))

        self.a = Parameter(torch.empty(self.out_features, **factory_kwargs))
        self.b = Parameter(torch.empty(self.out_features, **factory_kwargs))
        self.u0 = Parameter(torch.empty(self.out_features, **factory_kwargs), requires_grad=self.train_u0)

        self.reset_parameters()
        def step_fn(recurrent, alpha, beta, thr, a, b, u_rest, carry, cur):
            u_tm1, z_tm1, w_tm1 = carry
            if self.use_recurrent:
                cur_rec = F.linear(z_tm1, recurrent, None)
                cur = cur + cur_rec
            
            u = alpha * u_tm1 + (1.0 - alpha) * (
                cur - w_tm1
            )
            u_thr = u - thr
            z = spike_grad_injection_function(u_thr, self.alpha, self.c)
            u = u * (1 - z.detach()) + u_rest*z.detach()
            w = (
                beta * w_tm1 + (1.0 - beta) * (a * u_tm1 + b * z_tm1) * self.q
                )

            return (u, z, w), z
        self.step = step_fn
        
        def wrapped_scan(u0: Parameter, z0: Tensor, w0: Parameter, 
                         x: Tensor,
                         recurrent: Parameter, alpha: Parameter, beta: Parameter, 
                         thr: Tensor, a: Parameter, b: Parameter):
            if self.use_u_rest:
                u_rest = u0
            else:
                u_rest = torch.zeros_like(u0)
            def wrapped_step(carry, cur):
                return step_fn(recurrent, alpha, beta, thr, a, b, u_rest, carry, cur)

            return generic_scan(wrapped_step, (u0, z0, w0), x, self.unroll)
        def wrapped_scan_with_states(u0: Parameter, z0: Tensor, w0: Parameter, x: Tensor,
                         recurrent: Parameter, alpha: Parameter, beta: Parameter, 
                         thr: Tensor, a: Parameter, b: Parameter):
            if self.use_u_rest:
                u_rest = u0
            else:
                u_rest = torch.zeros_like(u0)
            def wrapped_step(carry, cur):
                return step_fn(recurrent, alpha, beta, thr, a, b, u_rest, carry, cur)

            return generic_scan_with_states(wrapped_step, (u0, z0, w0), x, self.unroll)
        self.wrapped_scan = wrapped_scan
        self.wrapped_scan_with_states = wrapped_scan_with_states
        if not cfg.get('compile', False):
            self.wrapped_scan = torch.compiler.disable(self.wrapped_scan, recursive=False)
            self.wrapped_scan_with_states = torch.compiler.disable(self.wrapped_scan_with_states, recursive=False)
        self.reset_parameters()
        
    def reset_parameters(self) -> None:
        self.tau_u_trainer.reset_parameters()
        self.tau_w_trainer.reset_parameters()
        
        
        torch.nn.init.uniform_(
            self.weight,
            -self.ff_gain * torch.sqrt(1 / torch.tensor(self.in_features)),
            self.ff_gain * torch.sqrt(1 / torch.tensor(self.in_features)),
        )
        
        torch.nn.init.zeros_(self.bias)
        
        # h0 states 
        if self.train_u0:
            torch.nn.init.uniform_(self.u0, 0, self.thr[0].item())
        else:
            torch.nn.init.zeros_(self.u0)
        if self.use_recurrent:
            torch.nn.init.orthogonal_(
                self.recurrent,
                gain=1.0,
            )
        
        torch.nn.init.uniform_(self.a, self.a_range[0], self.a_range[1])
        torch.nn.init.uniform_(self.b, self.b_range[0], self.b_range[1])
        
    def initial_state(self, batch_size:int, device: Optional[torch.device] = None) -> Tuple[Tensor, Tensor, Tensor]:
        size = (batch_size, self.out_features)
        z = torch.zeros(
            size=size, 
            device=device, 
            dtype=torch.float, 
            layout=None, 
            pin_memory=None,
            requires_grad=True
        )
        w = torch.zeros(
            size=size,
            device=device, 
            dtype=torch.float, 
            layout=None, 
            pin_memory=None,
            requires_grad=True
        )
        return self.u0.unsqueeze(0), z, w

    def apply_parameter_constraints(self):
        self.tau_u_trainer.apply_parameter_constraints()
        self.tau_w_trainer.apply_parameter_constraints()
        self.a.data = torch.clamp(self.a, min=self.a_range[0], max=self.a_range[1])
        self.b.data = torch.clamp(self.b, min=self.b_range[0], max=self.b_range[1])
        self.u0.data = self.u0 - torch.sign(self.u0)*torch.relu(torch.abs(self.u0) - self.thr)
        self.thr.data = torch.maximum(self.thr, torch.zeros_like(self.thr))
        
    def forward(
        self, input_tensor: Tensor,  states: Tuple[Tensor, Tensor, Tensor]
    ) -> Tuple[Tensor, Tensor]:
        decay_u = self.tau_u_trainer.get_decay()
        decay_w = self.tau_w_trainer.get_decay()
        soma_current = F.linear(input_tensor, self.weight, self.bias)
        new_states, z_t = self.step(self.recurrent, decay_u, decay_w, self.thr, self.a, self.b, self.u0, states, soma_current)
        
        return z_t, new_states
        
    def layer_forward(self, inputs: Tensor) -> Tensor:
        current = F.linear(inputs, self.weight, self.bias)
        decay_u = self.tau_u_trainer.get_decay()
        decay_w = self.tau_w_trainer.get_decay()
        u, z, w = self.initial_state(int(inputs.shape[0]), inputs.device)
        out_buffer = self.wrapped_scan(u, z, w, current, self.recurrent, decay_u, decay_w, self.thr, self.a, self.b)
        return out_buffer[:, :, :self.num_out_neuron]

    @torch.no_grad()
    def layer_forward_with_states(self, inputs) -> Tuple[Tensor, Tensor]:
        current = F.linear(inputs, self.weight, self.bias)
        decay_u = self.tau_u_trainer.get_decay()
        decay_w = self.tau_w_trainer.get_decay()
        u, z, w = self.initial_state(int(inputs.shape[0]), inputs.device)
        states, out = self.wrapped_scan_with_states(u, z, w, current, self.recurrent, decay_u, decay_w, self.thr, self.a, self.b)
        return states[..., :self.num_out_neuron], out[..., :self.num_out_neuron]
    
class SEAdLIF(EFAdLIF):
    def __init__(self, cfg, device=None, dtype=None, **kwargs):
        super().__init__(cfg, device, dtype, **kwargs)
        def step_fn(recurrent, alpha, beta, thr, a, b, u_rest, carry, cur):
            u_tm1, z_tm1, w_tm1 = carry
            if self.use_recurrent:
                cur_rec = F.linear(z_tm1, recurrent, None)
                cur = cur + cur_rec
            #膜电位更新
            #u_tm1膜电位的前一时刻值
            #cur (当前时刻的输入电流)
            #适应性电流的前一时刻值
            u = alpha * u_tm1 + (1.0 - alpha) * (
                cur - w_tm1
            )
            #脉冲生成
            u_thr = u - thr
            #调用替代梯度函数生成脉冲输出
            z = spike_grad_injection_function(u_thr, self.alpha, self.c)
            # 膜电位重制
            u = u * (1 - z.detach()) + u_rest*z.detach()
            # 自适应电流更新
            w = (
                beta * w_tm1 + (1.0 - beta) * (a * u + b * z) * self.q
                )
            return (u, z, w), z
        # 绑定时间步长函数
        self.step = step_fn

        def wrapped_scan(u0: Parameter, z0: Tensor, w0: Parameter, 
                          x: Tensor,
                         recurrent: Parameter, alpha: Parameter, beta: Parameter, 
                         thr: Tensor, a: Parameter, b: Parameter):
            # 静息电位设置
            if self.use_u_rest:
                u_rest = u0
            else:
                u_rest = torch.zeros_like(u0)
            # 包装时间步长函数
            def wrapped_step(carry, cur):
                return step_fn(recurrent, alpha, beta, thr, a, b, u_rest, carry, cur)
            return generic_scan(wrapped_step, (u0, z0, w0), x, self.unroll)
        def wrapped_scan_with_states(u0: Parameter, z0: Tensor, w0: Parameter, 
                          x: Tensor,
                         recurrent: Parameter, alpha: Parameter, beta: Parameter, 
                         thr: Tensor, a: Parameter, b: Parameter):
            if self.use_u_rest:
                u_rest = u0
            else:
                u_rest = torch.zeros_like(u0)
                
            def wrapped_step(carry, cur):
                return step_fn(recurrent, alpha, beta, thr, a, b, u_rest, carry, cur)
            return generic_scan_with_states(wrapped_step, (u0, z0, w0), x, self.unroll)
        self.wrapped_scan = wrapped_scan
        self.wrapped_scan_with_states = wrapped_scan_with_states
        if not cfg.get('compile', False):
            self.wrapped_scan = torch.compiler.disable(self.wrapped_scan, recursive=False)
            self.wrapped_scan_with_states = torch.compiler.disable(self.wrapped_scan_with_states, recursive=False)
        self.reset_parameters()