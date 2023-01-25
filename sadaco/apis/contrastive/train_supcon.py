from cProfile import label
from time import sleep
from typing import Callable, Optional, Union, Tuple, DefaultDict, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from torch.cuda.amp import autocast,GradScaler

from sadaco.utils.stats import print_stats

def move_device(data : DefaultDict, device : torch.device):
    return {k: d.to(device) for k,d in data.items() if hasattr(d, 'to')}
        
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
    grad_thres = None,
    update_interval = 1
) -> Optional[Union[DefaultDict, np.ndarray]]:
    model.train().to(device)
    train_losses = 0
    base_losses = 0
    contrast_losses = 0
    scaler = GradScaler()
    with tqdm(
        enumerate(train_loader),
        unit="batch",
        desc=f"Epoch [{epoch}]",
        total=train_loader.__len__(),
        leave=False
    ) as pbar:
        contrast_buffer = []
        label_buffers = {}
        for bidx, batch_info in pbar:
            if isinstance(batch_info, list):
                taglist = ['input', 'label', 'label2', 'lam', 'phase']
                batch_info = {k : v for k,v in zip(taglist, batch_info[:len(taglist)])}
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
                preprocessing.to(device)
                inputs = preprocessing(batch_info)
            else:
                inputs = batch_info
            with autocast():
                output, contrast_feats = model(inputs['input'])
                contrast_buffer.append(contrast_feats)
                for k, v in batch_info.items():
                    if 'label' in k:
                        if k in label_buffers.keys():
                            label_buffers[k].append(v)
                        else:
                            label_buffers[k] = [v]
                batch_info.update({'output':output, 'features' : torch.cat(contrast_buffer)})
                base_loss = base_criterion(**batch_info)
                batch_info.update({k : torch.cat(v) for k,v in label_buffers.items()})
                contrast_loss = contrast_criterion(**batch_info)
                loss = weights[0] * base_loss + weights[1] * contrast_loss
            scaler.scale(loss).backward()
            if (bidx+1) % update_interval == 0 or bidx == len(train_loader.dataset) - 1:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_thres)
                scaler.step(optimizer)
                scaler.update()
                model.zero_grad()
                optimizer.zero_grad(set_to_none=True)
                contrast_buffer = []
                for k in label_buffers.keys():
                    label_buffers[k] = []
            else:
                contrast_buffer[-1] = contrast_buffer[-1].detach()

            pbar.set_postfix(loss=loss.item())

            if base_criterion.reduction in ['mean', 'none']:
                train_losses += loss.item() * output.shape[0]
                base_losses += base_loss.item() * output.shape[0]
                contrast_losses += contrast_loss.item() * output.shape[0]
            else:
                train_losses += loss.item()
                base_losses += base_loss.item()
                contrast_losses += contrast_loss.item()
            
        train_losses /= len(train_loader.dataset)
        base_losses /= len(train_loader.dataset)
        contrast_losses /= len(train_loader.dataset)
        
        if verbose:
            print(print_stats((train_losses, base_losses, contrast_losses),
                              ('Total Loss', 'Cls Loss', 'Cont Loss')))
        if return_stats:
            return {'Total Loss' : train_losses, 'Cls Loss' : base_losses, 'Cont Loss' : contrast_losses}
        
        
        
def train_mixcon_epoch2(
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
    grad_thres = None,
    update_interval = 1,
    multi_label=False,
) -> Optional[Union[DefaultDict, np.ndarray]]:
    model.train().to(device)
    train_losses = 0
    base_losses = 0
    contrast_losses = 0
    scaler = GradScaler()
    with tqdm(
        enumerate(train_loader),
        unit="batch",
        desc=f"Epoch [{epoch}]",
        total=train_loader.__len__(),
        leave=False
    ) as pbar:
        contrast_buffer = []
        label_buffers = {}
        
        import torchaudio
        import torchvision
        
        fm = torchaudio.transforms.FrequencyMasking(freq_mask_param=10)
        tm = torchaudio.transforms.TimeMasking(time_mask_param=10)
        stretch = torchvision.transforms.Compose([
            torchvision.transforms.Resize((70, 280)),
            torchvision.transforms.RandomCrop((64, 256)),
            fm,
            tm
        ])
                
        for bidx, batch_info in pbar:
            if isinstance(batch_info, list):
                taglist = ['input', 'label', 'label2', 'lam', 'phase']
                batch_info = {k : v for k,v in zip(taglist, batch_info[:len(taglist)])}
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
                preprocessing.to(device)
                inputs = preprocessing(batch_info)
            else:
                inputs = batch_info
            with autocast():
                _input = inputs['input']
                _aug_input = torch.stack((_input, fm(_input), tm(_input), stretch(_input)), 1)
                _batch, _view, _channel, h, w = _aug_input.shape
                _aug_input = _aug_input.reshape(_batch*_view, _channel, h, w)
                lab1 = torch.stack((inputs['label'], inputs['label'],inputs['label'],inputs['label']), 1)
                lab1  = lab1.view(_batch*_view, -1)
                lab2 = torch.stack((inputs['label2'], inputs['label2'],inputs['label2'],inputs['label2']), 1)
                lab2  = lab2.view(_batch*_view, -1)
                ml = torch.cat((inputs['lam'], inputs['lam'],inputs['lam'],inputs['lam'],), 0)
                batch_info['input'] = _aug_input
                batch_info['label'] = lab1
                batch_info['label2'] = lab2
                batch_info['lam'] = ml
                
                output, contrast_feats = model(inputs['input'])
                contrast_buffer.append(contrast_feats)
                for k, v in batch_info.items():
                    if 'label' in k:
                        if k in label_buffers.keys():
                            label_buffers[k].append(v)
                        else:
                            label_buffers[k] = [v]
                batch_info.update({'output':output, 'features' : torch.cat(contrast_buffer)})
                base_loss = base_criterion(**batch_info)
                batch_info.update({k : torch.cat(v) for k,v in label_buffers.items()})
                
                batch_info['features'] = batch_info['features'].reshape(_batch, _view, -1)
                batch_info['label'] = lab1.view(_batch, _view, -1)[:,0,:]
                batch_info['label2'] = lab2.view(_batch, _view, -1)[:,0,:]
                batch_info['lam'] = ml.view(_batch, _view, -1)[:,0,:]
                
                if multi_label:
                    contrast_loss = 0
                    for k in range(output.shape[-1]):
                        contrast_loss += contrast_criterion(**{**batch_info, 'label':batch_info['label'][:,k], 'label2':batch_info['label2'][:,k]})
                    contrast_loss /= output.shape[-1]
                else:
                    contrast_loss = contrast_criterion(**batch_info)
                loss = weights[0] * base_loss + weights[1] * contrast_loss
            scaler.scale(loss).backward()
            if (bidx+1) % update_interval == 0 or bidx == len(train_loader.dataset) - 1:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_thres)
                scaler.step(optimizer)
                scaler.update()
                model.zero_grad()
                optimizer.zero_grad(set_to_none=True)
                contrast_buffer = []
                for k in label_buffers.keys():
                    label_buffers[k] = []
            else:
                contrast_buffer[-1] = contrast_buffer[-1].detach()

            pbar.set_postfix(loss=loss.item())

            if base_criterion.reduction in ['mean', 'none']:
                train_losses += loss.item() * output.shape[0]
                base_losses += base_loss.item() * output.shape[0]
                contrast_losses += contrast_loss.item() * output.shape[0]
            else:
                train_losses += loss.item()
                base_losses += base_loss.item()
                contrast_losses += contrast_loss.item()
            
        train_losses /= len(train_loader.dataset)
        base_losses /= len(train_loader.dataset)
        contrast_losses /= len(train_loader.dataset)
        
        if verbose:
            print(print_stats((train_losses, base_losses, contrast_losses),
                              ('Total Loss', 'Cls Loss', 'Cont Loss')))
        if return_stats:
            return {'Total Loss' : train_losses, 'Cls Loss' : base_losses, 'Cont Loss' : contrast_losses}