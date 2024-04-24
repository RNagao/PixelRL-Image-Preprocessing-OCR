import torch
from torch import Tensor
from typing import Iterable, Dict, Any

class ShareAdam(torch.optim.Adam):
    def __init__(self, params: Iterable[Tensor] | Iterable[Dict[str, Any]], lr: float | torch.Tensor = 0.001, betas: torch.Tuple[float] = (0.9, 0.99), eps: float = 1e-8, weight_decay: float = 0, amsgrad: bool = False, *, foreach: bool | None = None, maximize: bool = False, capturable: bool = False, differentiable: bool = False, fused: bool | None = None) -> None:
        super().__init__(params, lr, betas, eps, weight_decay, amsgrad, foreach=foreach, maximize=maximize, capturable=capturable, differentiable=differentiable, fused=fused)

        for group in self.param_groups:
            for p in group['params']:
                state = self.state['params']
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p.data)
                state['exp_avg_sq'] = torch.zeros_like(p.data)

                state['exp_avg'].share_memory_()
                state['exp_avg_sq'].share_memory_()