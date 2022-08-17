from time import sleep
from typing import Callable, Optional, Union, Tuple, DefaultDict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from torch.cuda.amp import autocast,GradScaler

def move_device(data : Tuple, device : torch.device):
    return (d.to(device) for d in data)
        
def train_basic_epoch(
    model: nn.Module,
    device: torch.device,
    train_loader: DataLoader,
    optimizer: torch.optim,
    criterion: Callable,
    epoch: int,
    return_stats: bool = False,
    verbose: bool = False,
    preprocessing : Callable = None,
) -> Optional[Union[DefaultDict, np.ndarray]]:
    model.train().to(device)

    train_loss = 0
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
                
                loss = criterion(output, *batch_info[1:])
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            pbar.set_postfix(loss=loss.item())

            train_loss += loss.item()
        train_loss /= len(train_loader.dataset)
        if verbose:
            print(f"Train: Average Loss: {train_loss:.3f}")
        if return_stats:
            return {'Loss' : train_loss}