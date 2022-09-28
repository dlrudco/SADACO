import datetime

from tqdm import tqdm

import torch

from sadaco.utils.config_parser import parse_config_obj
import torch

from sadaco.apis import models
from sadaco.utils.stats import print_stats
from torch.utils.data import DataLoader
from sadaco.pipelines.scheduler import BaseScheduler
import sadaco.dataman as dataman

    
from sadaco.dataman.loader import _build_dataloader

import sadaco.apis.losses as LF

class BaseTrainer():
    def __init__(self, train_configs):
        self.configs = train_configs
        self.data_configs = parse_config_obj(yml_path=self.configs.data_configs.file)
        self.model_configs = parse_config_obj(yml_path=self.configs.model_configs.file)

        ######## WANDB SETUP ########
        self.log_configs = {
                'K-Fold' : None,
                'Master' : {**self.configs.__dict__},
                'Dataset' : {**self.data_configs.__dict__},
                'Model' : {**self.model_configs.__dict__},
            }

        self.logger = self.build_logger(self.configs.use_wandb)
                
        self.model = self.build_model()
        # self.model = torch.hub.load('harritaylor/torchvggish', 'vggish')
        self.optimizer = self.build_optimizer()
        
        if 'train_dataset' not in self.__dict__.keys():
            self.build_dataset()
        if 'train_dataloader' not in self.__dict__.keys():
            self.build_dataloader()
        # self.build_dataloader(split='train')
        # self.build_dataloader(split='val')
        
        self.device = torch.device(
            f"cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model = self.model.to(self.device)
        
        self.train_criterion = build_criterion(self.configs.train.criterion.name, 
                                        mixup=self.configs.train.criterion.loss_mixup,
                                        **self.configs.train.criterion.params)
        self.valid_criterion = build_criterion(self.configs.train.criterion.name, mixup=False,
                                               **self.configs.train.criterion.params)
        self.scheduler = BaseScheduler(self.configs, self.optimizer, self.model, exp_id=self.logger.name, parallel=self.model_configs.data_parallel)
        self.preproc = None
        self._progress = -1
    
    def build_dataset(self):
        raise NotImplementedError
        self.train_dataset = None
        self.val_dataset = None
    
    def build_dataloader(self):
        if self.configs.data.train_dataloader.sampler is not None:
            train_sampler = getattr(dataman, self.configs.data.train_dataloader.sampler.name)(
                                    self.train_dataset, **self.configs.data.train_dataloader.sampler.params)
        else:
            train_sampler = None
        
        if train_sampler is not None:
            for _key in ['shuffle', 'batch_size', 'drop_last', 'sampler']:
                try:
                    self.configs.data.train_dataloader.params.pop(_key)
                except KeyError:
                    pass
        self.train_loader = DataLoader(self.train_dataset, batch_sampler=train_sampler, **self.configs.data.train_dataloader.params)
        self.val_loader = DataLoader(self.val_dataset, **self.configs.data.val_dataloader.params)
        
    def build_model(self):
        model = models.build_model(self.model_configs)
        return model

    def parallel(self):
        if self.model_configs.data_parallel == True:
            self.model = torch.nn.DataParallel(self.model)    
        
    def build_optimizer(self, trainables = None):
        if trainables is None:
            trainables = [p for p in self.model.parameters() if p.requires_grad]
        else:
            pass
        optimizer = getattr(torch.optim, self.configs.train.optimizer.name)(
            trainables, **self.configs.train.optimizer.params)
        return optimizer
    
    def build_logger(self, use_wandb=True):
        if self.configs.fold is not None and f'{self.configs.fold}-Fold' not in self.configs.prefix:
            self.configs.prefix = f'{self.configs.fold}-Fold_{self.configs.prefix}'
        if use_wandb:
            try: 
                import wandb
                group_id = self.configs.prefix
                now = datetime.datetime.now()
                exp_id = now.strftime('%Y_%m_%d_') + wandb.util.generate_id()
                logger = wandb.init(project=self.configs.project_name, group=group_id, name=exp_id, config=self.log_configs, entity='sadaco')
                self.use_wandb = True
            except ModuleNotFoundError:
                from apis.logger import BaseLogger
                logger = BaseLogger(config=self.log_configs)
                self.use_wandb = False
        else:
            from apis.logger import BaseLogger
            logger = BaseLogger(config=self.log_configs)
            self.use_wandb = False
        return logger
    
    def reset_trainer(self):
        '''
        TODO : Currently calls __init__ again
            which might cause unexpected behavior
            find a way to aviod this.
        '''
        if self.use_wandb:
            self.logger.finish(quiet=True)
        else:
            pass
        self.__init__(self.configs)
        
    def resume(self):
        if self.model_configs.resume is not None and self.configs.model_configs.resume:
            checkpoint = torch.load(self.model_configs.resume)
            self.model.load_state_dict(checkpoint['state_dict'])
            if self.configs.model_configs.resume_optimizer:
                self.optimizer.load_state_dict(checkpoint['opts'])
            else:
                pass
        else:
            pass
        
    def train(self):    
        for epoch in tqdm(range(self.configs.train.max_epochs)):
            self._progress = epoch
            train_stats = self.train_epoch(epoch)
            valid_stats = self.validate_epoch(epoch)
            if self.scheduler is not None:
                self.scheduler.step(train_stats, valid_stats)
                valid_stats.update({f'Best/{k}' : v for k,v in self.scheduler.best_all_stats.items()})
            self.logger.log({**train_stats, **valid_stats})
        return 0
    
    def train_kfold(self, k):
        self.log_configs['K-Fold'] = k
        for i in tqdm(range(k)):
            self.prepare_kfold(i, k)
            for epoch in tqdm(range(self.configs.train.max_epochs), leave=False):
                self._progress = epoch
                train_stats = self.train_epoch(epoch)
                valid_stats = self.validate_epoch(epoch)
                if self.scheduler is not None:
                    self.scheduler.step(train_stats, valid_stats)
                    valid_stats.update({f'Best/{k}' : v for k,v in self.scheduler.best_all_stats.items()})
                self.logger.log({**train_stats, **valid_stats})
            self.reset_trainer()
        self.build_dataset()
        self.build_dataloader()
        return 0
        
    def prepare_kfold(self, i, k):
        import random
        from random import randint
        import math
        
        if i == 0:
            if self.configs.seed in [None, -1]:
                self.seed = randint(100000, 999999)
            else:
                self.seed = self.configs.seed
            self._kfold_original = {'train' : {'data' : self.train_dataset.data, 'label':self.train_dataset.labels},
                                    'val' : {'data' : self.val_dataset.data, 'label':self.val_dataset.labels}}
            self._kfold_data_list = self.train_dataset.data + self.val_dataset.data
            self._kfold_label_list = self.train_dataset.labels + self.val_dataset.labels
            data_num = len(self._kfold_data_list)
            index_list = list(range(data_num))
            random.seed(self.seed)
            random.shuffle(index_list)
            self._kfold_index_chunks = []
            self._kfold_sizes = []
            self._kfold_chunk_size = math.ceil(data_num / k)
            for n in range(k):
                try:
                    self._kfold_index_chunks.append(index_list[(n)*self._kfold_chunk_size:(n+1)*self._kfold_chunk_size])
                except IndexError:
                    self._kfold_index_chunks.append(index_list[(n)*self._kfold_chunk_size:])
                self._kfold_sizes.append(len(self._kfold_index_chunks[n]))
            print(self._kfold_sizes)
        else:
            pass
        val_list = []
        train_list = []
        for n in range(k):
            if n == i:
                val_list.extend(self._kfold_index_chunks[n])
            else:
                train_list.extend(self._kfold_index_chunks[n])
        self.train_dataset.data = [self._kfold_data_list[j] for j in train_list]
        self.train_dataset.labels = [self._kfold_label_list[j] for j in train_list]
        self.val_dataset.data = [self._kfold_data_list[j] for j in val_list]
        self.val_dataset.labels = [self._kfold_label_list[j] for j in val_list]
        self.build_dataloader()
    
    def validate(self, return_stats=True):
        valid_stats = self.validate_epoch(0)
        print(print_stats(valid_stats))
        if return_stats:
            return valid_stats
        return 0
    
    def test(self, **kwargs):
        self.validate(**kwargs)

    def train_epoch(self):
        raise NotImplementedError
    
    def validate_epoch(self):
        raise NotImplementedError
    
    
def build_optimizer(model, train_configs, trainables = None):
    if trainables is None:
        trainables = [p for p in model.parameters() if p.requires_grad]
    else:
        pass
    optimizer = getattr(torch.nn.optim, train_configs.optimizer.name)(trainables, **train_configs.optimizer.params)
    return optimizer
    
def build_dataloader(dataset, train_configs, data_configs):
    loader = _build_dataloader(dataset, train_configs, data_configs)
    return loader

def build_criterion(name, mixup=False, **kwargs):
    criterion = getattr(LF, name)
    if mixup : 
        criterion = LF.mixup_criterion(criterion, **kwargs)
    else:
        criterion = criterion(**kwargs)
    return criterion

if __name__ == "__main__":
    print("This file contains base trainer modules. Main call is under preparation")
    print("-- DEBUG --")
    breakpoint()