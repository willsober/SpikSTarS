from functools import partial

from typing import Callable, Tuple, Any
import torch
import math

# SLAYER surrogate gradient function
def SLAYER(x: torch.Tensor, alpha: float, c: float) -> torch.Tensor:
    return c * alpha / (2 * torch.exp(x.abs() * alpha))

def spike_grad_injection_function(x: torch.Tensor, alpha: float, c: float) -> torch.Tensor:
    # Forward Gradient Injection trick (credits to Sebastian Otte)
    return torch.heaviside(x, torch.as_tensor(0.0).type(x.dtype)).detach() + (x - x.detach()) * SLAYER(x, alpha, c).detach()
def generic_scan(
    f: Callable[[Tuple[torch.Tensor, ...], torch.Tensor], Tuple[Tuple[torch.Tensor, ...], torch.Tensor]],  # f(s_t, x) -> (s_t+1, y)
    init: Tuple[torch.Tensor, ...],
    xs: torch.Tensor,
    unroll: int = 1,
) -> torch.Tensor:


    """
        Create a scan like procedure that can be optimized by torch.compile.
        Code was lifted from https://github.com/pytorch/pytorch/issues/50688#issuecomment-2315002649 (SamPruden)
        
        What the code do:
        This is a pseudo scan function https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.scan.html
        
        A scan in a higher-order function that loops over a statefull function from an initial state and an input list:
        
        def scan(f, init, xs):
            carry = init
            ys = []
            for x in xs:
                carry, y = f(carry, x)
                ys.append(y)
            return carry, np.stack(ys)
        
        A scan has the possibility to unroll in such a way that 
        that k iterations of f are made inside the loop instead of 1.
        
        This effectively creates two loops.
        def scan(f, init, xs, unroll=1):
            if unroll == 1: # do normal scan
            
            num_chunk = math.ceil(xs/unroll) 
            xs_chunk = np.split_array(xs, num_chunk)
            
            carry = init
            ys = []
            for chunk in xs_chunk: # outer loop
                y_chunk =[]
                for x in chunk: # inner loop
                    carry, y = f(carry, x)
                y_chunk.append(y)
                ys.extends(y_chunk)
            return carry, np.stack(ys)
        
        In this code, only the inner loop is compiled, the outer loop is kept in non-optimised code.
        The code is written so that ys is buffered.
        
         
        Reasoning:
        By default torch.compile will fully unroll each loop in the computation flow.
        This is not ideal for RNN where the number of iterations in the loop can be large.
        As such, the intermediate representation may require a large amount of virtual registers that cannot be matched to the hardware registers. (I assume, I'm not a low level guy.) 
        Also, unrolling means more instructions (larger binary) need to be stored to the device. 
        This (again, I assume) can lead to less room for data and memory spills/cache misses, slowing down the computation instead of improving it.
        Unrolling is generally a good thing (better cache locality, pipelining, out-of-order execution, reduced loop overhead), 
        but ideally the depth of unrolling should match the CPU/GPU capacity.

        Args:
            f (Callable[[tuple[torch.Tensor, ...], torch.Tensor], tuple[tuple[torch.Tensor, ...], torch.Tensor]]) step function
            init (tuple[torch.Tensor, ...]): intial carry/states
            xs (torch.Tensor): inputs tensor
            unroll (int, optional): unrolling factor. Defaults to 1.

        Returns:
            torch.Tensor: output tensor
        """
    init_carry = init
    num_chunk = math.ceil(xs.shape[1] / unroll)
    out_ys = torch.empty_like(xs)
    
    # def unrolled_body_(local_carry: Tuple[torch.Tensor, ...], xs: torch.Tensor, local_out_ys: torch.Tensor):
    #
    #     for i in range(xs.shape[1]):
    #         print(f"[DEBUG] y.shape: {y.shape}")
    #
    #         local_carry, y = f(local_carry, xs[:, i])
    #         local_out_ys[:, i] = y[:, 0]
    #
    #
    #         # local_out_ys[:, i] = y.squeeze(1)
    #     return local_carry
    def unrolled_body_(local_carry: Tuple[torch.Tensor, ...], xs: torch.Tensor, local_out_ys: torch.Tensor):
        for i in range(xs.shape[1]):
            try:
                local_carry, y = f(local_carry, xs[:, i])

                # print(f"local_out_ys[:, {i}].shape = {local_out_ys[:, i].shape}")

                # y = y[:, 0, :, :]  # 取出第一个'something'，得到 [64, 151, 64]
                # print(f"nihao1.shape before squeeze: {y.shape}")
                # # y = y[:, 0, :, :]
                # print(f"nihao2.shape before squeeze: {y.shape}")
                local_out_ys[:, i] = y  # 已在 wrapped_step 里 squeeze 了
            except Exception as e:
                print(f"[ERROR] Step {i} failed: {e}")
                raise
        return local_carry

    @partial(torch.compiler.disable, recursive = False)
    def do_uncompiled_loop():
        carry = init_carry
        for i in range(num_chunk):
            carry = unrolled_body_(carry, xs[:, i * unroll:][:, :unroll], out_ys[:, i * unroll:][:, :unroll])

    do_uncompiled_loop()
    return out_ys

def generic_scan_with_states(
    f: Callable[[Tuple[torch.Tensor, ...], torch.Tensor], Tuple[Tuple[torch.Tensor, ...], torch.Tensor]], # f(s_t, x) -> (s_t+1, y)
    init: Tuple[torch.Tensor, ...],
    xs: torch.Tensor,
    unroll: int = 1,
) -> Tuple[Tuple[torch.Tensor, ...], torch.Tensor]:
    """ 
    Same logic that generic scan but return states.
    Only used for visualization purpose and should not be used with grad mode
    """
    num_chunk = math.ceil(xs.shape[1] / unroll)
    out_ys = torch.empty_like(xs)
    carry_out = torch.stack([torch.concat((x.unsqueeze(1).expand((xs.shape[0], 1, -1)), torch.empty_like(xs)), dim=1) for x in init], dim=0)
    def unrolled_body_(local_carry_out: Tuple[torch.Tensor, ...], xs: torch.Tensor, local_out_ys: torch.Tensor):
        local_carry = local_carry_out[:, :, 0].unbind(0)
        for i in range(xs.shape[1]):
            local_carry, y = f(local_carry, xs[:, i])
            local_carry_out[:, :, i+1] = torch.stack(local_carry, 0)
            local_out_ys[:, i] = y
            
    @partial(torch.compiler.disable, recursive = False)
    def do_uncompiled_loop():
        for i in range(num_chunk):
            unrolled_body_(carry_out[:, :, i*unroll:, :][:, :, :unroll + 1, :], xs[:, i * unroll:][:, :unroll], out_ys[:, i * unroll:][:, :unroll])
    do_uncompiled_loop()
    return carry_out, out_ys

def A_law(x: torch.Tensor, a: float = 87.6):
    sign_x = torch.sign(x)
    abs_x = torch.abs(x)
    log_a = torch.log(a)
    y1 =  (a*abs_x)/(1 + log_a)
    y2 = (1 + torch.log(abs_x) + log_a)/(1 + log_a)
    y = torch.where(abs_x < 1/a, y1, y2)
    return sign_x*y

def inverse_A_law(y: torch.Tensor, a: float = 87.6):
    sign_y = torch.sign(y)
    abs_y = torch.abs(y)
    log_a_p1 = torch.log(a) + 1
    x1 = (abs_y*log_a_p1)/a
    x2 = torch.exp(-1 + abs_y*log_a_p1)/a
    x = torch.where(abs_y < 1/log_a_p1, x1, x2)    
    return sign_y*x