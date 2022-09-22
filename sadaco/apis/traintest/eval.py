from time import sleep
from typing import Callable, Optional, Union, DefaultDict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from sadaco.utils.stats import ICBHI_Metrics, print_stats
from torch.cuda.amp import autocast,GradScaler

def move_device(data : DefaultDict, device : torch.device):
    return {k: d.to(device) for k,d in data.items() if hasattr(d, 'to')}

def test_basic_epoch(
    model: nn.Module,
    device: torch.device,
    test_loader: DataLoader,
    metrics: ICBHI_Metrics,
    criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    epoch: int,
    verbose: bool = True,
    preprocessing : Callable = None,
)-> Optional[Union[DefaultDict, np.ndarray]]:
    model.eval().to(device)
    test_loss = 0
    with tqdm(
        enumerate(test_loader),
        unit="batch",
        desc=f"Epoch [{epoch}]",
        total=test_loader.__len__(),
        leave=False
    ) as pbar, torch.no_grad(), autocast():
        for bidx, batch_info in pbar:
            if isinstance(batch_info, list):
                taglist = ['input', 'label', 'label2', 'lam', 'phase']
                batch_info = {k : v for k,v in zip(taglist, batch_info[:len(taglist)])}
            batch_info = move_device(batch_info, device)
            keep_info = batch_info['input']
            if torch.isnan(batch_info['input']).any():
                print(f'NaN mag!!! val before preproc')
            if preprocessing is not None:
                preprocessing.to(device)
                inputs = preprocessing(batch_info)
            else:
                inputs = batch_info
            if torch.isnan(inputs['input']).any():
                print(f'NaN mag!!! val after preproc')
            output = model(inputs['input'])
            batch_info.update({'output':output})
            loss = criterion(**batch_info).mean()
            if torch.isnan(loss):
                breakpoint()
            if criterion.reduction in ['mean', 'none'] :
                test_loss += loss.item() * output.shape[0]
            else:
                test_loss += loss.item()
            pbar.set_postfix(loss=loss.item()/(bidx+1))
            metrics.update_lists(logits=output, y_true=torch.argmax(batch_info['label'], dim=-1))

    test_loss /= test_loader.dataset.__len__()
    stats = metrics.get_stats()
    stats.update({'Test Loss' : test_loss})
    metrics.reset_metrics()
    if verbose:
        print(print_stats(stats))
    return stats