import torch
import torch.nn
from typing import Union
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss, MSELoss

class CELoss(CrossEntropyLoss):
    """_summary_

    :param CrossEntropyLoss: _description_
    :type CrossEntropyLoss: _type_
    """    
    def __init__(self, mode : Union[str, int] ='onehot', **kwargs):
        """_summary_

        :param mode: _description_, defaults to 'onehot'
        :type mode: Union[str, int], optional
        :raises ValueError: _description_
        """        
        super().__init__(**kwargs)
        self.base_forward = super().forward
        
        if mode in ['onehot', 0]:
            self.mode = 0
        elif mode in ['int', 1]:
            self.mode = 1
        else:
            raise ValueError("Currently only Supporting One-hot or Integer")
        
    def forward(self, output, label, **kwargs):
        """_summary_

        :param output: _description_
        :type output: _type_
        :param label: _description_
        :type label: _type_
        :return: _description_
        :rtype: _type_
        """        
        if self.mode == 0:
            target = torch.argmax(label, axis=-1)
        else :
            target = target
        return self.base_forward(output, target)
    
    
class BCEWithLogitsLoss(BCEWithLogitsLoss):
    """_summary_

    :param BCEWithLogitsLoss: _description_
    :type BCEWithLogitsLoss: _type_
    """    
    def __init__(self, mode : Union[str, int] ='multihot', max=None, **kwargs):
        """_summary_

        :param mode: _description_, defaults to 'multihot'
        :type mode: Union[str, int], optional
        :param max: _description_, defaults to None
        :type max: _type_, optional
        :raises ValueError: _description_
        :raises ValueError: _description_
        """        
        super().__init__(**kwargs)
        self.base_forward = super().forward
        
        if mode in ['multihot', 0]:
            self.mode = 0
        else:
            raise ValueError("Currently only Supporting Multi-hot")
        
    def forward(self, output:torch.Tensor, label:torch.Tensor, **kwargs):
        """_summary_

        :param input: _description_
        :type input: torch.Tensor
        :param label: _description_
        :type label: torch.Tensor
        :return: _description_
        :rtype: _type_
        """        
        if self.mode == 0:
            target = label
        else :
            raise ValueError("Currently only Supporting Multi-hot")
        return self.base_forward(output, target)


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
