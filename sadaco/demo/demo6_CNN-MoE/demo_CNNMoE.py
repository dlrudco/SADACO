import datetime
import os
import uuid

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

import apis.models as models
from dataman.fraiwan.loader import FraiwanDataset
from pipeline import train_epoch, test_epoch
from utils.config_parser import parse_config_obj
from utils.misc import seed_everything
from utils.stats import Evaluation_Metrics


TRAIN_CONFIGS_YML = "configs_/train.yml"
DATA_CONFIGS_YML = "configs_/data.yml"
MODEL_CONFIGS_YML = "configs_/model.yml"


def main(train_configs_yml: str, data_configs_yml: str, model_configs_yml: str):
    train_configs = parse_config_obj(yml_path=train_configs_yml)
    data_configs = parse_config_obj(yml_path=data_configs_yml)
    model_configs = parse_config_obj(yml_path=model_configs_yml)

    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_name: str = (
        train_configs.prefix
        + datetime.datetime.now().strftime("_%Y%m%d_")
        + uuid.uuid1().hex[-5:]
    )
    log_dir: str = os.path.join(train_configs.save_stats_dir, run_name)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    seed_everything(seed=train_configs.seed)

    labels = pd.read_csv(data_configs.label_file, sep="\t", header=None)
    train, test = train_test_split(
        labels,
        test_size=data_configs.test_size,
        random_state=train_configs.seed if train_configs.seed != -1 else None,
    )
    train_dataset = FraiwanDataset(
        data_list=train[0].apply(lambda name: name + ".wav").to_list(),
        label_list=train[1].to_list(),
        data_dir=data_configs.data_dir,
        train=True,
        output_dim=data_configs.output_dim,
        get_weights=train_configs.weighted_criterion,
        sample_rate=data_configs.sample_rate,
        nfft=data_configs.nfft,
        hop_length=data_configs.hop_length,
        time_mask=data_configs.time_mask,
    )
    test_dataset = FraiwanDataset(
        data_list=test[0].apply(lambda name: name + ".wav").to_list(),
        label_list=test[1].to_list(),
        data_dir=data_configs.data_dir,
        train=False,
        output_dim=data_configs.output_dim,
        get_weights=train_configs.weighted_criterion,
        sample_rate=data_configs.sample_rate,
        nfft=data_configs.nfft,
        hop_length=data_configs.hop_length,
    )

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=data_configs.batch_size,
        shuffle=True,
        num_workers=data_configs.num_workers,
        pin_memory=True,
    )
    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=data_configs.batch_size,
        shuffle=False,
        num_workers=data_configs.num_workers,
        pin_memory=True,
    )

    train_metrics = Evaluation_Metrics(
        num_classes=model_configs.num_classes,
        normal_class_label=data_configs.normal_class_label,
    )
    test_metrics = Evaluation_Metrics(
        num_classes=model_configs.num_classes,
        normal_class_label=data_configs.normal_class_label,
    )

    model = getattr(models, model_configs.model_name)(**model_configs)

    train_criterion = getattr(torch.nn, train_configs.criterion)(
        weight=train_dataset.normed_weights.to(device),
        reduction="none" if data_configs.mixup_alpha > 0.0 else "mean",
    )

    test_criterion = getattr(torch.nn, train_configs.criterion)()

    if model_configs.from_checkpoint is not None:
        state_dict = torch.load(
            model_configs.from_checkpoint,
            map_location=device,
        )
        model.load_state_dict(state_dict)
        print(f"===> Loaded model from {model_configs.from_checkpoint}.")

    model.to(device)

    optimizer = getattr(torch.optim, train_configs.optimizer)(
        model.parameters(),
        lr=train_configs.lr,
    )
    scheduler = getattr(torch.optim.lr_scheduler, train_configs.lr_scheduler)(
        optimizer, **train_configs.lr_scheduler_params
    )

    train_stats = []
    test_stats = []
    for epoch in range(1, train_configs.max_epochs + 1):
        train_result = train_epoch(
            model=model,
            device=device,
            train_loader=train_dataloader,
            optimizer=optimizer,
            metrics=train_metrics,
            criterion=train_criterion,
            mixup=True if data_configs.mixup_alpha > 0.0 else False,
            epoch=epoch,
            return_stats=train_configs.save_stats,
        )
        if train_configs.save_stats:
            train_stats.append(train_result)
        if epoch % train_configs.val_interval == 0:
            test_result = test_epoch(
                model=model,
                device=device,
                test_loader=test_dataloader,
                metrics=test_metrics,
                criterion=test_criterion,
            )
            test_stats.append(test_result)
        scheduler.step()

    print(f"===> Training finished. Saving model and stats to {log_dir}.")

    np.save(os.path.join(log_dir, "train_stats.npy"), np.array(train_stats))
    np.save(os.path.join(log_dir, "test_stats.npy"), np.array(test_stats))


if __name__ == "__main__":
    main(
        train_configs_yml=TRAIN_CONFIGS_YML,
        data_configs_yml=DATA_CONFIGS_YML,
        model_configs_yml=MODEL_CONFIGS_YML,
    )
