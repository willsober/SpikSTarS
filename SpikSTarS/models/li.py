from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch.nn import Module
from torch import Tensor
from torch.nn.parameter import Parameter

from models.helpers import generic_scan, generic_scan_with_states
from module.tau_trainers import TauTrainer, get_tau_trainer_class
from omegaconf import DictConfig

class LI(Module):
    __constants__ = ["in_features", "out_features"]
    in_features: int
    out_features: int
    weight: torch.Tensor

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
        self.train_tau_u_method = cfg.get('train_tau_u_method', 'fixed')
        self.ff_gain = cfg.get('ff_gain', 1.0)
        
        
        self.weight = Parameter(
            torch.empty((self.out_features, self.in_features), **factory_kwargs)
        )

        self.bias = Parameter(torch.empty(self.out_features, **factory_kwargs))
        self.tau_u_trainer: TauTrainer = get_tau_trainer_class(self.train_tau_u_method)(
            self.out_features,
            self.dt,
            self.tau_u_range[0],
            self.tau_u_range[1],
            **factory_kwargs,
        )
        self.unroll = cfg.get('unroll', 10)
        def step_fn(alpha, carry, x):
            u, = carry
            u = alpha * u + (1.0 - alpha)*x
            return (u,), u 
        self.step = step_fn
        def wrapped_scan(u0: Parameter, x: Tensor, alpha: Parameter):
            def wrapped_step(u0, x):
                return step_fn(alpha, u0, x)
            return generic_scan(wrapped_step, (u0, ), x, self.unroll)
        
        def wrapped_scan_with_states(u0: Parameter, x: Tensor, alpha: Parameter):
            def wrapped_step(u0, x):
                return step_fn(alpha, u0, x)
            return generic_scan_with_states(wrapped_step, (u0,), x, self.unroll)
        self.wrapped_scan = wrapped_scan
        if not cfg.get('compile', False):
            self.wrapped_scan = torch.compiler.disable(self.wrapped_scan, recursive=False)
        #TODO: do not work properly in compile mode due to 1D tuple shape
        self.wrapped_scan_with_states = torch.compiler.disable(wrapped_scan_with_states, recursive=False)        
        self.reset_parameters()
    
    def reset_parameters(self):
        self.tau_u_trainer.reset_parameters()
        torch.nn.init.uniform_(
            self.weight,
            -self.ff_gain  * torch.sqrt(1 / torch.tensor(self.in_features)),
            self.ff_gain * torch.sqrt(1 / torch.tensor(self.in_features)),
        )
        torch.nn.init.zeros_(self.bias)

    @torch.compiler.disable
    def extra_repr(self) -> str:
        return "in_features={}, out_features={}, bias={}".format(
            self.in_features, self.out_features, self.bias is not None
        )
        
    def initial_state(
        self, batch_size: int, device: Optional[torch.device] = None
    ) -> tuple[torch.Tensor,]:
        size = (batch_size, self.out_features)
        u = torch.zeros(size=size, 
            device=device, 
            dtype=torch.float, 
            layout=None, 
            pin_memory=None,
            requires_grad=True
        )
        return (u,)

    def forward(self, input_tensor: Tensor, states: Tensor) -> Tuple[Tensor, Tensor]:
        decay_u = self.tau_u_trainer.get_decay()
        current = F.linear(input_tensor, self.weight, self.bias)
        new_states, u_t = self.step(decay_u, states, current)
        return u_t, new_states
     
    def layer_forward(self, inputs):
        current = F.linear(inputs, self.weight, self.bias)
        u, = self.initial_state(inputs.shape[0], device=inputs.device)
        decay_u = self.tau_u_trainer.get_decay()
        return self.wrapped_scan(u, current, decay_u)

    @torch.no_grad()
    def layer_forward_with_states(self, inputs) -> Tuple[Tensor, Tensor]:
        current = F.linear(inputs, self.weight, self.bias)
        u, = self.initial_state(inputs.shape[0], device=inputs.device)
        decay_u = self.tau_u_trainer.get_decay()
        states, out = self.wrapped_scan_with_states(u, current, decay_u)
        return states, out
    
    def apply_parameter_constraints(self):
        self.tau_u_trainer.apply_parameter_constraints()