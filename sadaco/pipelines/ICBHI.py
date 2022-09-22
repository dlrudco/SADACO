from turtle import update
from sadaco.apis.traintest import BaseTrainer,  train_basic_epoch, test_basic_epoch
from sadaco.apis.contrastive import ContrastTrainer, train_mixcon_epoch
from sadaco.apis.traintest import preprocessings as preps
from sadaco.utils.stats import ICBHI_Metrics, print_stats
from sadaco.utils.config_parser import ArgsParser
from build_modules import build_criterion
from scheduler import BaseScheduler

from sadaco.dataman.icbhi.dummy import RespiDatasetSTFT

import torch
# torch.autograd.set_detect_anomaly(True)
from torch.utils.data import DataLoader

class ICBHI_Basic_Trainer(BaseTrainer):
    def __init__(self, configs):
        super().__init__(configs)
        self.resume()
        
        self.preproc = preps.Preprocessor(
                            [preps.stft2meldb(n_stft=self.train_dataset.n_stft, n_mels = self.train_dataset.num_mel, 
                                              sample_rate=self.train_dataset.sample_rate)]
                            )
        self.evaluator = ICBHI_Metrics(num_classes=4, normal_class_label=0)
        breakpoint()
        
    def build_dataset(self):
        self.train_dataset = RespiDatasetSTFT(split='train', **self.data_configs.train)
        self.val_dataset = RespiDatasetSTFT(split='val', **self.data_configs.val)
    
    def train_epoch(self, epoch):
        train_stats = train_basic_epoch(model=self.model, device=self.device, train_loader=self.train_loader, 
                                        optimizer=self.optimizer,criterion=self.train_criterion, epoch=epoch, 
                                        return_stats=True, verbose = False, preprocessing=self.preproc, grad_thres=10., update_interval=self.configs.train.update_interval)
        return train_stats
    
    def validate_epoch(self, epoch):
        val_stats = test_basic_epoch(self.model,self.device, self.val_loader, self.evaluator,
                        criterion=self.valid_criterion, epoch=epoch, verbose=False, preprocessing=self.preproc)
        return val_stats
    
    
class ICBHI_Contrast_Trainer(ContrastTrainer):
    def __init__(self, configs):
        super().__init__(configs)
        self.preproc = preps.Preprocessor(
                    [preps.stft2meldb(n_stft=self.train_dataset.n_stft, n_mels = self.train_dataset.num_mel, 
                                      sample_rate=self.train_dataset.sample_rate)]
                    )
        self.attach_extractor()
        self.wrap_model()
        
        # Should redo optimizer building since the model is wrapped
        self.optimizer = self.build_optimizer()
        self.resume()
        self.contrast_criterion = build_criterion(self.configs.train.contrast_criterion.name, 
                                                  mixup=self.configs.train.contrast_criterion.loss_mixup)
        self.train_criterion = build_criterion(self.configs.train.criterion.name, 
                                        mixup=self.configs.train.criterion.loss_mixup,
                                        **self.configs.train.criterion.params)
        self.valid_criterion = build_criterion(self.configs.train.criterion.name, mixup=False,
                                               **self.configs.train.criterion.params)
        self.scheduler = BaseScheduler(self.configs, self.optimizer, self.model, exp_id=self.logger.name, parallel=self.model_configs.data_parallel)
        self.evaluator = ICBHI_Metrics(num_classes=4, normal_class_label=0)
        
        
    def build_dataset(self):
        self.train_dataset = RespiDatasetSTFT(split='train', **self.data_configs.train)
        self.val_dataset = RespiDatasetSTFT(split='val', **self.data_configs.val)
    
    def train_epoch(self, epoch):
        train_stats = train_mixcon_epoch(model=self.model, device=self.device, train_loader=self.train_loader, 
                                        optimizer=self.optimizer, base_criterion=self.train_criterion, 
                                        contrast_criterion=self.contrast_criterion, epoch=epoch, 
                                        return_stats=True, verbose = False, preprocessing = self.preproc,
                                        grad_thres=10., update_interval=self.configs.train.update_interval)
        return train_stats

    def validate_epoch(self, epoch):
        val_stats = test_basic_epoch(self.model,self.device, self.val_loader, self.evaluator,
                        criterion=self.valid_criterion, epoch=epoch, verbose=False, preprocessing=self.preproc)
        return val_stats
        
    
def main(configs):
    if configs.train.method == 'contrastive':
        trainer = ICBHI_Contrast_Trainer(configs)
    elif configs.train.method == 'basic':
        trainer = ICBHI_Basic_Trainer(configs)
    else:
        raise ValueError("Method is not on the available list of [basic, contrastive]")
    if configs.fold is None:
        trainer.train()
    else:
        trainer.train_kfold(configs.fold)
    # results = trainer.test(return_stats=True)
    # print(print_stats(results))

def parse_configs():
    parser = ArgsParser()
    # One can use the config files for the default settings,
    # and override settings by manually giving the arguments
    # Currently, overriding only the top-level arguments are available
    parser.add_argument("--mixup", action='store_true')
    parser.add_argument("--fold", default=None, type=int)
    parser.add_argument("--seed", default=None, type=int)
    args = parser.get_args()
    return args

if __name__ == "__main__":
    configs = parse_configs()
    from sadaco.utils.misc import seed_everything
    seed_everything(configs.seed)
    main(configs)