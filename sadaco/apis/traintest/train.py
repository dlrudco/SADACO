from time import sleep
from typing import Callable, Optional, Union, Tuple, DefaultDict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from torch.cuda.amp import autocast,GradScaler

def move_device(data : DefaultDict, device : torch.device):
    return {k: d.to(device) for k,d in data.items() if hasattr(d, 'to')}
        
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
    grad_thres = None,
    update_interval = 1
) -> Optional[Union[DefaultDict, np.ndarray]]:
    """_summary_

    :param model: _description_
    :type model: nn.Module
    :param device: _description_
    :type device: torch.device
    :param train_loader: _description_
    :type train_loader: DataLoader
    :param optimizer: _description_
    :type optimizer: torch.optim
    :param criterion: _description_
    :type criterion: Callable
    :param epoch: _description_
    :type epoch: int
    :param return_stats: _description_, defaults to False
    :type return_stats: bool, optional
    :param verbose: _description_, defaults to False
    :type verbose: bool, optional
    :param preprocessing: _description_, defaults to None
    :type preprocessing: Callable, optional
    :param grad_thres: _description_, defaults to None
    :type grad_thres: _type_, optional
    :param update_interval: _description_, defaults to 1
    :type update_interval: int, optional
    :return: _description_
    :rtype: Optional[Union[DefaultDict, np.ndarray]]
    """    
    model.train().to(device)

    train_loss = 0
    scaler = GradScaler()
    with tqdm(
        enumerate(train_loader),
        unit="batch",
        desc=f"Epoch [{epoch}]",
        total=train_loader.__len__(),
        leave=False
    ) as pbar:
        model.zero_grad()
        optimizer.zero_grad(set_to_none=True)
        for bidx, batch_info in pbar:
            if isinstance(batch_info, list):
                taglist = ['input', 'label', 'label2', 'lam', 'phase']
                batch_info = {k : v for k,v in zip(taglist, batch_info[:len(taglist)])}
            batch_info = move_device(batch_info, device)
            
            '''
            The first element of the tuple 'batch_info' should be the 
            input tensor.
            And the rest should be used in the criterion
            [INPUT, LABEL, (opt)LABEL2, ...]
            '''
            if preprocessing is not None:
                preprocessing.to(device)
                inputs = preprocessing(batch_info)
                if torch.isnan(inputs['input']).any():
                    print(f'NaN mag!!! after preproc')
            else:
                inputs = batch_info
            with autocast():
                output = model(inputs['input'])
                if torch.isnan(output).any():
                    print(f'NaN mag!!! after model')
                batch_info.update({'output':output})
                loss = criterion(**batch_info)
            if criterion.reduction in ['mean', 'none']:
                scaler.scale(loss).backward()
            else:
                scaler.scale(loss).backward()
                
            if (bidx+1) % update_interval == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_thres)
                scaler.step(optimizer)
                scaler.update()
                model.zero_grad()
                optimizer.zero_grad(set_to_none=True)
            else:
                pass
            if criterion.reduction in ['mean', 'none']:
                train_loss += loss.item() * output.shape[0]
            else:
                train_loss += loss.item()
            pbar.set_postfix(loss=train_loss/(bidx+1))

            
        train_loss /= len(train_loader.dataset)
        if verbose:
            print(f"Train: Average Loss: {train_loss:.3f}")
        if return_stats:
            return {'Train Loss' : train_loss}