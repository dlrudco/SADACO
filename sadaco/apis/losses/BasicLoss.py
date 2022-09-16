import torch
from typing import Union


class CELoss(torch.nn.CrossEntropyLoss):
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
    
    
class BCEWithLogitsLoss(torch.nn.BCEWithLogitsLoss):
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
        
    def forward(self, input:torch.Tensor, label:torch.Tensor, **kwargs)->torch.Tensor:
        if self.mode == 0:
            target = label
        else :
            temptar = torch.zeros(label.shape[0], max).to(label.device)
            temptar[:,label] = 1.
            target = temptar
        return self.base_forward(input, target)