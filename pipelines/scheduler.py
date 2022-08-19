import os
import torch
import math
import shutil
from torch.optim.lr_scheduler import _LRScheduler

class CosineAnnealingWarmUpRestarts(_LRScheduler):
    def __init__(self, optimizer, T_0, T_mult=1, eta_max=0.1, T_up=0, gamma=1., last_epoch=-1, verbose=False):
        if T_0 <= 0 or not isinstance(T_0, int):
            raise ValueError("Expected positive integer T_0, but got {}".format(T_0))
        if T_mult < 1 or not isinstance(T_mult, int):
            raise ValueError("Expected integer T_mult >= 1, but got {}".format(T_mult))
        if T_up < 0 or not isinstance(T_up, int):
            raise ValueError("Expected positive integer T_up, but got {}".format(T_up))
        self.T_0 = T_0
        self.T_mult = T_mult
        self.base_eta_max = eta_max
        self.eta_max = eta_max
        self.T_up = T_up
        self.T_i = T_0
        self.gamma = gamma
        self.cycle = 0
        self.T_cur = last_epoch
        super(CosineAnnealingWarmUpRestarts, self).__init__(optimizer, last_epoch, verbose)
    
    def get_lr(self):
        if self.T_cur == -1:
            return self.base_lrs
        elif self.T_cur < self.T_up:
            return [(self.eta_max - base_lr)*self.T_cur / self.T_up + base_lr for base_lr in self.base_lrs]
        else:
            return [base_lr + (self.eta_max - base_lr) * (1 + math.cos(math.pi * (self.T_cur-self.T_up) / (self.T_i - self.T_up))) / 2
                    for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.T_cur = self.T_cur + 1
            if self.T_cur >= self.T_i:
                self.cycle += 1
                self.T_cur = self.T_cur - self.T_i
                self.T_i = (self.T_i - self.T_up) * self.T_mult + self.T_up
        else:
            if epoch >= self.T_0:
                if self.T_mult == 1:
                    self.T_cur = epoch % self.T_0
                    self.cycle = epoch // self.T_0
                else:
                    n = int(math.log((epoch / self.T_0 * (self.T_mult - 1) + 1), self.T_mult))
                    self.cycle = n
                    self.T_cur = epoch - self.T_0 * (self.T_mult ** n - 1) / (self.T_mult - 1)
                    self.T_i = self.T_0 * self.T_mult ** (n)
            else:
                self.T_i = self.T_0
                self.T_cur = epoch
                
        self.eta_max = self.base_eta_max * (self.gamma**self.cycle)
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr
    
    
class BaseScheduler():
    def __init__(self, train_configs, optimizer, model, exp_id=None):
        self.metric = train_configs.train.target_metric
        if self.metric is None:
            self.metric = 'Accuracy'
        
        if train_configs.train.lr_scheduler is not None:
            try:
                self.lr_scheduler = getattr(torch.optim.lr_scheduler, train_configs.train.lr_scheduler.name)(
                optimizer, **train_configs.train.lr_scheduler.params
                )
            except AttributeError:
                self.lr_scheduler = globals()[train_configs.train.lr_scheduler.name](
                optimizer, **train_configs.train.lr_scheduler.params
                )
        self.model = model
        self.optimizer=optimizer
        self.configs = train_configs
        self.best_score = 0
        self.epoch = 0
        self.save_interval = self.configs.train.save_interval
        if exp_id is None:
            self.exp_id = self.configs.prefix
        else:
            self.exp_id = os.path.join(self.configs.prefix, exp_id)
        os.makedirs(os.path.join(self.configs.output_dir, self.exp_id), exist_ok=True)
        
    def step(self, train_stats, valid_stats, *lr_sched_args):
        if self.lr_scheduler is not None:
            self.lr_scheduler.step(*lr_sched_args)
        
        if (self.epoch + 1) % max(self.save_interval, 1) == 0:
            filename=os.path.join(self.configs.output_dir, self.exp_id, 'checkpoint.pth')
            state = {
                        'epoch': self.epoch + 1,
                        'state_dict': self.model.state_dict(),
                        'best_val_acc': self.best_score,
                        'opts' : self.optimizer.state_dict(),
                        }
            torch.save(state, filename)
            is_best =  self.best_score <= valid_stats[self.metric]
            if is_best:
                # print("\nSave new best model\n")
                self.best_score = valid_stats[self.metric]
                shutil.copyfile(filename, os.path.join(self.configs.output_dir, self.exp_id, 'checkpoint_best.pth'))
        
        self.epoch += 1