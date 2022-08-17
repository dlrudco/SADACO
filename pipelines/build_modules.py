import torch
from dataman.loader import _build_dataloader

def build_optimizer(model, train_configs, trainables = None):
    if trainables is None:
        trainables = [p for p in model.parameters() if p.requires_grad]
    else:
        pass
    optimizer = getattr(torch.nn.optim, train_configs.optimizer.name)(trainables, **train_configs.optimizer.params)
    return optimizer
    
def build_dataloader(dataset, train_configs, data_configs):
    loader = _build_dataloader(dataset, train_configs, data_configs)
    return loader