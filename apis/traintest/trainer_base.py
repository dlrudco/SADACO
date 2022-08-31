import datetime

from tqdm import tqdm

import torch

from utils.config_parser import parse_config_obj
import torch

from apis import models
from utils.stats import print_stats

class BaseTrainer():
    def __init__(self, train_configs):
        self.configs = train_configs
        self.data_configs = parse_config_obj(yml_path=self.configs.data_configs.file)
        self.model_configs = parse_config_obj(yml_path=self.configs.model_configs.file)

        ######## WANDB SETUP ########
        self.log_configs = {
                **self.configs.__dict__,
                **self.data_configs.__dict__,
                **self.model_configs.__dict__,
            }
        self.logger = self.build_logger(self.configs.use_wandb)
        self.model = self.build_model()
        # self.model = torch.hub.load('harritaylor/torchvggish', 'vggish')
        self.optimizer = self.build_optimizer()
        
        self.build_dataset()
        self.build_dataloader()
        # self.build_dataloader(split='train')
        # self.build_dataloader(split='val')
        
        self.device = torch.device(
            f"cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model = self.model.to(self.device)
        self.scheduler = None
    
    def build_dataset(self):
        raise NotImplementedError
    
    def build_dataloader(self):
        raise NotImplementedError
    
    def build_model(self):
        model = models.build_model(self.model_configs)
        if self.model_configs.data_parallel == True:
            model = torch.nn.DataParallel(model)
        return model
    
    def build_optimizer(self, trainables = None):
        if trainables is None:
            trainables = [p for p in self.model.parameters() if p.requires_grad]
        else:
            pass
        optimizer = getattr(torch.optim, self.configs.train.optimizer.name)(
            trainables, **self.configs.train.optimizer.params)
        return optimizer
    
    def build_logger(self, use_wandb=True):
        if use_wandb:
            try: 
                import wandb
                group_id = self.configs.prefix
                now = datetime.datetime.now()
                exp_id = now.strftime('%Y_%m_%d_') + wandb.util.generate_id()
                logger = wandb.init(project=self.configs.project_name, group=group_id, name=exp_id, config=self.log_configs, entity='sadaco')
            except ModuleNotFoundError:
                from apis.logger import BaseLogger
                logger = BaseLogger(config=self.log_configs)
        else:
            from apis.logger import BaseLogger
            logger = BaseLogger(config=self.log_configs)
        return logger
    
    def train(self):    
        for epoch in tqdm(range(self.configs.train.max_epochs)):
            train_stats = self.train_epoch(epoch)
            valid_stats = self.validate_epoch(epoch)
            if self.scheduler is not None:
                self.scheduler.step(train_stats, valid_stats)
                valid_stats.update({f'Best/{self.scheduler.metric}' : self.scheduler.best_score})
            self.logger.log({**train_stats, **valid_stats})
    
    def validate(self, return_stats=True):
        valid_stats = self.validate_epoch(0)
        print(print_stats(valid_stats))
        if return_stats:
            return valid_stats
    
    def test(self, **kwargs):
        self.validate(**kwargs)

    def train_epoch(self):
        raise NotImplementedError
    
    def validate_epoch(self):
        raise NotImplementedError