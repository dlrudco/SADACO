import datetime
import os
import uuid
from typing import Union, Any

import wandb
from munch import DefaultMunch

# from torch.utils.tensorboard import SummaryWriter


class LoggerMan:
    r"""LoggerMan is a wrapper class for the wandb and tensorboardX libraries.
    User may choose to log to wandb or tensorboardX.

    Agrs:
        train_configs: Training configuration munch class.
        data_configs: Data configuration munch class.
        model_configs: Model_configuration munch class.
    """

    def __init__(
        self,
        train_configs: DefaultMunch,
        data_configs: DefaultMunch,
        model_configs: DefaultMunch,
    ) -> Union[wandb.run, Any]:
        all_configs = {
            **train_configs.__dict__,
            **data_configs.__dict__,
            **model_configs.__dict__,
        }
        self.train_configs = train_configs
        self.data_configs = data_configs
        self.model_configs = model_configs

        run_name: str = (
            train_configs.prefix
            + datetime.datetime.now().strftime("_%Y%m%d_")
            + uuid.uuid1().hex[-5:]
        )  # TODO: modify for k-fold
        log_dir: str = os.path.join(self.train_configs.save_stats_dir, run_name, "logs")

        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        if self.train_configs.wandb:
            self.logger = wandb.init(
                project=self.train_configs.project_name,
                name=run_name,
                config=all_configs,
                dir=log_dir,
                tags=list(self.train_configs.wandb_tags),
            )
        # else:
        #     self.logger = SummaryWriter(
        #         log_dir=log_dir
        #     )
        #     self.logger.add_hparams(hparam_dict=all_configs)

    def watch_(self, **kwargs: Any) -> None:
        if not self.train_configs.wandb:
            raise ValueError("Tensorboard does not support watch function.")
        return wandb.watch(**kwargs)
