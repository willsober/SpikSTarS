
from typing import Optional, Sequence

import torch._dynamo.guards
from models.helpers import generic_scan, generic_scan_with_states, spike_grad_injection_function
import torch
import torch.nn.functional as F
from torch.nn import Module
from torch import Tensor
from torch.nn.parameter import Parameter

from module.tau_trainers import TauTrainer, get_tau_trainer_class
from omegaconf import DictConfig
class LIF(Module):
    __constants__ = ["in_features", "out_features"]
    in_features: int
    out_features: int
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
        self.dt = cfg.get('dt', 1.0)
        self.tau_u_range = cfg.tau_u_range
        self.train_tau = cfg.get('train_tau', 'interpolation')
        self.unroll = cfg.get('unroll', 10)
        self.use_recurrent = cfg.get('use_recurrent', True)
        self.ff_gain = cfg.get('ff_gain', 1.0)
        thr = cfg.get('thr', 1.0)
        self.num_out_neuron = cfg.get('num_out_neuron', self.out_features)
        self.use_u_rest = cfg.get('use_u_rest', False)
        self.train_u0 = cfg.get('train_u0', False)
        if isinstance(thr, Sequence):
            thr = torch.FloatTensor(self.out_features, device=device).uniform_(thr[0], thr[1])
        else:
            thr = torch.Tensor([thr,])
        if cfg.get('train_thr', False):
            self.thr = Parameter(thr)
        else:
            self.register_buffer('thr', thr)
            
        self.alpha = cfg.get('alpha', 5.0)
        self.c = cfg.get('c', 0.4)

        self.weight = Parameter(
            torch.empty((self.out_features, self.in_features), **factory_kwargs)
        )
        self.bias = Parameter(torch.empty(self.out_features, **factory_kwargs))
        if self.use_recurrent:
            self.recurrent = Parameter(
                    torch.empty((self.out_features, self.out_features), **factory_kwargs)
                )
        else:
            # registering an empty size tensor is required for the static analyser
            self.register_buffer("recurrent", torch.empty(size=()))
        self.tau_u_trainer: TauTrainer = get_tau_trainer_class(self.train_tau)(
            self.out_features,
            self.dt,
            self.tau_u_range[0],
            self.tau_u_range[1],
            **factory_kwargs,
        )
        self.u0 = Parameter(torch.empty(self.out_features, **factory_kwargs), requires_grad=self.train_u0)
        
        self.reset_parameters()
        def step_fn(recurrent, alpha, thr, u_rest, carry, cur):
            u_tm1, z_tm1 = carry
            if self.use_recurrent:
                cur_rec = F.linear(z_tm1, recurrent, None)
                cur = cur + cur_rec
            
            u = alpha * u_tm1 + (1.0 - alpha) * (
                cur 
            )
            u_thr = u - thr
            z = spike_grad_injection_function(u_thr, self.alpha, self.c)
            u = u * (1 - z.detach()) + u_rest*z.detach()
            return (u, z), z
        self.step = step_fn
        
        def wrapped_scan(u0: Parameter,
                         z0: Tensor, x: Tensor,
                         recurrent: Parameter, alpha: Parameter,
                         thr: Tensor):
            if self.use_u_rest:
                u_rest = u0
            else:
                u_rest = torch.zeros_like(u0)
                
            def wrapped_step(carry, cur):
                return step_fn(recurrent, alpha, thr, u_rest, carry, cur)

            return generic_scan(wrapped_step, (u0, z0), x, self.unroll)
        def wrapped_scan_with_states(u0: Parameter,
                         z0: Tensor, x: Tensor,
                         recurrent: Parameter, alpha: Parameter,
                         thr: Tensor):
            if self.use_u_rest:
                u_rest = u0
            else:
                u_rest = torch.zeros_like(u0)
            def wrapped_step(carry, cur):
                return step_fn(recurrent, alpha, thr, u_rest, carry, cur)

            return generic_scan_with_states(wrapped_step, (u0, z0), x, self.unroll)
        self.wrapped_scan = wrapped_scan
        self.wrapped_scan_with_states = wrapped_scan_with_states

    def reset_parameters(self):
        self.tau_u_trainer.reset_parameters()
        torch.nn.init.uniform_(
            self.weight,
            -self.ff_gain * torch.sqrt(1 / torch.tensor(self.in_features)),
            self.ff_gain * torch.sqrt(1 / torch.tensor(self.in_features)),
        )
        torch.nn.init.zeros_(self.bias)
        if self.use_recurrent:
            torch.nn.init.orthogonal_(
                self.recurrent,
                gain=1.0,
            )
        # h0 states 
        if self.train_u0:
            torch.nn.init.uniform_(self.u0, 0, self.thr[0].item())
        else:
            torch.nn.init.zeros_(self.u0)
    def initial_state(
        self, batch_size: int, device: Optional[torch.device] = None
    ) -> tuple[Tensor, Tensor]:
        size = (batch_size, self.out_features)
        z = torch.zeros(size=size, 
                        device=device, 
                        dtype=torch.float,
                        layout=None, 
                        pin_memory=None
                        )
        return self.u0.unsqueeze(0), z
    
    def forward(self, input_tensor: Tensor, states: tuple[Tensor, Tensor]) -> tuple[Tensor, Tensor]:
        decay_u = self.tau_u_trainer.get_decay()
        soma_current = F.linear(input_tensor, self.weight, self.bias)
        new_states, z_t = self.step(self.recurrent, decay_u, self.thr, self.u0,  states, soma_current)
        return z_t, new_states

    def layer_forward(self, inputs: torch.Tensor) -> Tensor:
        current = F.linear(inputs, self.weight, self.bias)
        decay_u = self.tau_u_trainer.get_decay()
        u, z = self.initial_state(inputs.shape[0], inputs.device)
        out_buffer = self.wrapped_scan(u, z, current, self.recurrent, decay_u, self.thr)
        return out_buffer[:, :, :self.num_out_neuron]
    
    @torch.no_grad()
    def layer_forward_with_states(self, inputs: torch.Tensor) -> Tensor:
        current = F.linear(inputs, self.weight, self.bias)
        decay_u = self.tau_u_trainer.get_decay()
        u, z = self.initial_state(inputs.shape[0], inputs.device)
        states, out_buffer = self.wrapped_scan_with_states(u, z, current, self.recurrent, decay_u, self.thr)
        return states[..., :self.num_out_neuron], out_buffer[..., :self.num_out_neuron]
    def apply_parameter_constraints(self):
        self.tau_u_trainer.apply_parameter_constraints()
        self.u0.data = self.u0 - torch.sign(self.u0)*torch.relu(torch.abs(self.u0) - self.thr)
        self.thr.data = torch.maximum(self.thr, torch.zeros_like(self.thr))