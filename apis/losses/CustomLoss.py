import torch
from typing import Union, Callable, Optional, Any

def mixup_criterion(criterion: Callable, **criterion_options):
    class MixedCriterion(criterion):
        def __init__(self, **criterion_options):
            self.reduction = None
            super().__init__(**criterion_options)
            if self.reduction is None:
                if 'reduction' in criterion_options.keys():
                    self.reduction = criterion_options['reduction']
                else:
                    self.reduction = 'mean'
            
        def __call__(self, output: torch.Tensor, label: torch.Tensor, label2: torch.Tensor,
                    lam: Union[float, torch.Tensor], **kwargs) -> torch.Tensor:
            mixup_loss = lam * self.forward(output=output, label=label, **kwargs) + (1 - lam) * self.forward(output=output, label=label2, **kwargs)
            if self.reduction == 'sum':
                mixup_loss = mixup_loss.sum()
            else:
                mixup_loss = mixup_loss.mean()
            return mixup_loss
        
    return MixedCriterion(**criterion_options)