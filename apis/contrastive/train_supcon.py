from time import sleep
from typing import Callable, Optional, Union, Tuple, DefaultDict, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from torch.cuda.amp import autocast,GradScaler

from utils.stats import print_stats

def move_device(data : Tuple, device : torch.device):
    return (d.to(device) for d in data)
        
def train_mixcon_epoch(
    model: nn.Module,
    device: torch.device,
    train_loader: DataLoader,
    optimizer: torch.optim,
    base_criterion: Callable,
    contrast_criterion: Callable,
    epoch: int,
    weights : List[int] = [1, 1],
    return_stats: bool = True,
    verbose: bool = False,
    preprocessing : Callable = None,
) -> Optional[Union[DefaultDict, np.ndarray]]:
    model.train().to(device)
    train_losses = 0
    base_losses = 0
    contrast_losses = 0
    scaler = GradScaler()
    with tqdm(
        train_loader,
        unit="batch",
        desc=f"Epoch [{epoch}]",
        total=train_loader.__len__(),
    ) as pbar:
        for batch_info in pbar:
            batch_info = move_device(batch_info, device)
            
            model.zero_grad()
            optimizer.zero_grad(set_to_none=True)
            '''
            The first element of the tuple 'batch_info' should be the 
            input tensor.
            And the rest should be used in the criterion
            [INPUT, LABEL, (opt)LABEL2, ...]
            '''
            if preprocessing is not None:
                inputs = preprocessing(*batch_info)
            else:
                inputs = batch_info[0]
            with autocast():
                output = model(inputs)
                layer_name = model.handler.layer_name[0]
                contrast_feats = model.handler.get_features(layer_name).to(device)
                base_loss = base_criterion(output, *batch_info[1:])
                contrast_loss = contrast_criterion(output, contrast_feats, *batch_info[1:])
                loss = weights[0] * base_loss + weights[1] * contrast_loss
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            pbar.set_postfix(loss=loss.item())

            train_loss += loss.item()
            base_losses += base_loss.item()
            contrast_losses += contrast_loss.item()
        train_loss /= len(train_loader.dataset)
        base_losses /= len(train_loader.dataset)
        contrast_losses /= len(train_loader.dataset)
        
        if verbose:
            print(print_stats((train_loss, base_losses, contrast_losses),
                              ('Total Loss', 'Cls Loss', 'Cont Loss')))
        if return_stats:
            return {'Total Loss' : train_loss, 'Cls Loss' : base_losses, 'Cont Loss' : contrast_losses}