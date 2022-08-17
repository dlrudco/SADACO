import torch
from typing import Union, Callable, Optional, Any

def mixup_criterion(criterion: Callable, **criterion_options):
    class MixedCriterion(criterion):
        def __init__(self, **criterion_options):
            super().__init__(**criterion_options)
            
            
        def __call__(self, pred: torch.Tensor, y_a: torch.Tensor, y_b: torch.Tensor,
                    lam: Union[float, torch.Tensor]) -> torch.Tensor:
            mixup_loss = lam * self.forward(pred, y_a) + (1 - lam) * self.forward(pred, y_b)
            return mixup_loss
        
    return MixedCriterion(**criterion_options)