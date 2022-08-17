from time import sleep
from typing import Callable, Optional, Union, DefaultDict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.stats import ICBHI_Metrics, print_stats
from torch.cuda.amp import autocast,GradScaler

def move_device(data : Tuple, device : torch.device):
    return (d.to(device) for d in data)

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
        test_loader,
        unit="batch",
        desc=f"Epoch [{epoch}]",
        total=test_loader.__len__(),
    ) as pbar, torch.no_grad(), autocast():
        for batch_info in pbar:
            batch_info = move_device(batch_info, device)
            if preprocessing is not None:
                inputs = preprocessing(*batch_info)
            else:
                inputs = batch_info[0]
            output = model(inputs)
            test_loss += criterion(output, *batch_info[1:]).item()
            metrics.update_lists(logits=output, y_true=batch_info[1])

    test_loss /= len(test_loader.dataset)
    stats = metrics.get_stats()
    metrics.reset_metrics()
    if verbose:
        print(print_stats(stats))
    return stats