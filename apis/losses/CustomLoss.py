import torch
from typing import Union, Callable, Optional, Any

def mixup_criterion(criterion: Callable, **criterion_options):
    class MixedCriterion(criterion):
        def __init__(self, **criterion_options):
            super().__init__(**criterion_options)
            
            
            
        def __call__(self, output: torch.Tensor, label: torch.Tensor, label2: torch.Tensor,
                    lam: Union[float, torch.Tensor], **kwargs) -> torch.Tensor:
            mixup_loss = lam * self.forward(output=output, label=label) + (1 - lam) * self.forward(output=output, label=label2)
            if self.reduction == 'sum':
                mixup_loss = mixup_loss.sum()
            else:
                mixup_loss = mixup_loss.mean()
            return mixup_loss
        
    return MixedCriterion(**criterion_options)