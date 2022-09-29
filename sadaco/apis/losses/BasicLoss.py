import torch
import torch.nn
from typing import Union
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss, MSELoss

class CELoss(CrossEntropyLoss):
    def __init__(self, mode : Union[str, int] ='onehot', **kwargs):
        super().__init__(**kwargs)
        self.base_forward = super().forward
        
        if mode in ['onehot', 0]:
            self.mode = 0
        elif mode in ['int', 1]:
            self.mode = 1
        else:
            raise ValueError("Currently only Supporting One-hot or Integer")
        
    def forward(self, output, label, **kwargs):
        if self.mode == 0:
            target = torch.argmax(label, axis=-1)
        else :
            target = target
        return self.base_forward(output, target)
    
    
class BCEWithLogitsLoss(BCEWithLogitsLoss):
    """A modified version of the BCEWithLogitsLoss

    :param BCEWithLogitsLoss: Parent Loss fn from pytorch
    :type BCEWithLogitsLoss: _type_
    """    
    def __init__(self, mode : Union[str, int] ='multihot', max=None, **kwargs):
        super().__init__(**kwargs)
        self.base_forward = super().forward
        
        if mode in ['multihot', 0]:
            self.mode = 0
        elif mode in ['int', 1]:
            self.mode = 1
            if max is None:
                raise ValueError("The number of Maximum Class should be provided, but given 'None' instead")
        else:
            raise ValueError("Currently only Supporting Multi-hot or Integer")
        
    def forward(self, input:torch.Tensor, label:torch.Tensor, **kwargs):
        if self.mode == 0:
            target = label
        else :
            temptar = torch.zeros(label.shape[0], max).to(label.device)
            temptar[:,label] = 1.
            target = temptar
        return self.base_forward(input, target)


class Normalized_MSELoss(MSELoss):
    """A modified version of the MSELoss for non-constrastive self-supervised learning in BYOL, which is between the normalized predictions and target projections.

    :param MSELoss: Parent Loss 'MSELoss' from pytorch
    :type MSELoss: Loss fn
    :return: MSE Loss value
    :rtype: torch.Tensor
    """ 

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.base_forward = super().forward

    def forward(
        self, predictions: torch.Tensor, target_projections: torch.Tensor, **kwargs
    ):
        normalized_predictions = torch.nn.functional.normalize(predictions, dim=-1, p=2)
        normalized_target = torch.nn.functional.normalize(
            target_projections, dim=-1, p=2
        )
        return self.base_forward(normalized_predictions, normalized_target)
