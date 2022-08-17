from time import sleep
from typing import Callable, Optional, Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.stats import Evaluation_Metrics
from torch.cuda.amp import autocast,GradScaler

def test_epoch(
    model: nn.Module,
    device: torch.device,
    test_loader: DataLoader,
    metrics: Evaluation_Metrics,
    criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    verbose: bool = True,
) -> np.ndarray:
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            metrics.update_lists(logits=output, y_true=target)

    test_loss /= len(test_loader.dataset)
    acc, se, sp, sc = metrics.get_stats()
    metrics.reset_metrics()
    if verbose:
        print(
            f"Test: Average Loss: {test_loss:.3f} Acc: {acc:.3f}, SE: {se:.3f}, SP: {sp:.3f}, SC: {sc:.3f}"
        )
    return np.array([test_loss, acc, se, sp, sc])