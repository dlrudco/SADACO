from time import sleep
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.stats import Evaluation_Metrics


def mixup_criterion(
    criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    pred: torch.Tensor,
    y_a: torch.Tensor,
    y_b: torch.Tensor,
    lam: float,
) -> torch.Tensor:
    mixup_loss = lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
    return mixup_loss.mean()


def train_epoch(
    model: nn.Module,
    device: torch.device,
    train_loader: DataLoader,
    optimizer: torch.optim,
    metrics: Evaluation_Metrics,
    criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    mixup: bool,
    epoch: int,
    return_stats: bool = False,
    verbose: bool = False,
) -> Optional[np.ndarray]:
    model.train()
    train_loss = 0
    with tqdm(
        train_loader,
        unit="batch",
        desc=f"Epoch [{epoch}]",
        total=train_loader.__len__(),
    ) as tepoch:
        if mixup:
            for data, target_1, target_2, mixup_lambda in tepoch:
                data, target_1, target_2, mixup_lambda = (
                    data.to(device),
                    target_1.to(device),
                    target_2.to(device),
                    mixup_lambda.to(device),
                )
                optimizer.zero_grad(set_to_none=True)
                output = model(data)
                loss = mixup_criterion(
                    criterion, output, target_1, target_2, mixup_lambda
                )
                loss.backward()
                optimizer.step()
                metrics.update_mixup_stats(
                    logits=output,
                    y_true_a=target_1,
                    y_true_b=target_2,
                    mixup_lambda=mixup_lambda,
                )
                tepoch.set_postfix(loss=loss.item())
                sleep(0.1)
                train_loss += loss.item()
            train_loss /= len(train_loader.dataset)
            acc = metrics.get_mixup_stats()
            metrics.reset_metrics()
            if verbose:
                print(f"Train: Average Loss: {train_loss:.3f} Acc: {acc:.3f}")
            if return_stats:
                return np.array([train_loss, acc])
        else:
            for data, target in tepoch:
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad(set_to_none=True)
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                metrics.update_lists(logits=output, y_true=target)
                tepoch.set_postfix(loss=loss.item())
                sleep(0.1)
                train_loss += loss.item()
            train_loss /= len(train_loader.dataset)
            acc, se, sp, sc = metrics.get_stats()
            metrics.reset_metrics()
            if verbose:
                print(
                    f"Train: Average Loss: {train_loss:.3f} Acc: {acc:.3f}, SE: {se:.3f}, SP: {sp:.3f}, SC: {sc:.3f}"
                )
            if return_stats:
                return np.array([train_loss, acc, se, sp, sc])


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
