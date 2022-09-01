from apis.traintest import BaseTrainer
import torch

class ContrastTrainer(BaseTrainer):
    def __init__(self, train_configs):
        super().__init__(train_configs)
        
    def attach_extractor(self):
        target_layer = self.model_configs.contrastive_layer
        
        self.mapper = 
