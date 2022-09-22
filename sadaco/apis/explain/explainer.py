import torch
from .hookman import FGHandler

class BaseExplainer(FGHandler): 
    def __init__(self, model, layers=None):
        super().__init__(model, layer_name = layers)
        
class GradcamExplainer(BaseExplainer):
    def __init__(self, model, layers):
        super().__init__(model, layers)
        
    def forward(self, inputs):
        pass