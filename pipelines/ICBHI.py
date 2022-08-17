from apis.traintest import BaseTrainer, train_basic_epoch, test_basic_epoch
from apis.contrastive import train_mixcon_epoch
from apis.traintest import preprocessings as preps
from utils.stats import print_stats
from utils.config_parser import ArgsParser
from .build_modules import build_criterion
from .scheduler import BaseScheduler

class ICBHI_Basic_Trainer(BaseTrainer):
    def __init__(self, configs):
        super().__init__(configs)
        self.criterion = build_criterion(self.configs.criterion, mixup=self.configs.criterion.loss_mixup)
        self.preproc = preps.Preprocessor(
                            [preps.stft2meldb(n_stft=self.train_dataset.n_stft),
                             preps.normalize(mean=self.train_dataset.norm_mean, std = self.train_dataset.norm_std * 2)]
                            )
        self.scheduler = BaseScheduler(self.configs, self.optimizer, self.model, exp_id=self.logger.exp_id)
        
    def build_dataset(self):
        raise NotImplementedError
    
    def build_dataloader(self):
        raise NotImplementedError
    
    def train_epoch(self, epoch):
        train_stats = train_basic_epoch(model=self.model, device=self.device, train_loader=self.train_loader, 
                                        optimizer=self.optimizer,criterion=self.criterion, epoch=epoch, 
                                        return_stats=True, verbose = False, preprocessing=self.preproc)
        self.logger.log(train_stats)
        return train_stats
    
    def validate_epoch(self):
        raise NotImplementedError
    
    
class ICBHI_Contrast_Trainer(ICBHI_Basic_Trainer):
    def __init__(self, configs):
        super().__init__(configs)
        self.criterion = build_criterion(self.configs.criterion, mixup=self.configs.criterion.loss_mixup)
        self.contrast_criterion = build_criterion(self.configs.contrast_criterion, mixup=self.configs.contrast_criterion.loss_mixup)
        
    def train_epoch(self, epoch):
        train_stats = train_mixcon_epoch(model=self.model, device=self.device, train_loader=self.train_loader, 
                                        optimizer=self.optimizer,base_criterion=self.criterion, 
                                        contrast_criterion=self.contrast_criterion, epoch=epoch, 
                                        return_stats=True, verbose = False, preprocessing = self.preproc)
        self.logger.log(train_stats)
        return train_stats
        
        
def main(configs):
    if configs.contrastive:
        trainer = ICBHI_Contrast_Trainer(configs)
    else:
        trainer = ICBHI_Basic_Trainer(configs)
    trainer.train()
    results = trainer.test(return_stats=True)
    print(print_stats(results))

def parse_configs():
    parser = ArgsParser()
    # One can use the config files for the default settings,
    # and override settings by manually giving the arguments
    # Currently, overriding only the top-level arguments are available
    parser.add_argument("--mixup", action='store_true')
    args = parser.get_args()
    return args

if __name__ == "__main__":
    configs = parse_configs()
    main(configs)