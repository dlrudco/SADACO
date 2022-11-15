from typing import Dict
from sadaco.dataman.base import BaseDataset

import torch
import torchaudio
import torchvision

import os
import random
import numpy as np
from tqdm import tqdm

class icbhi_dataset(BaseDataset):
    def __init__(self, configs, split='train', transforms:Dict=None):
        super().__init__(configs, split)
        
        if 'mixup' in configs.__dict__.keys():
            self.mixup = configs.mixup.do
            self.mixup_rate = configs.mixup.rate
        else:
            self.mixup = False
            self.mixup_rate = 0.
        self.fixed_length = configs.output_length
        self.transforms = transforms
        if configs.num_label is not None:
            self.num_label = int(configs.num_label)
        else:
            self.num_label = 3
        assert self.num_label in [1,2,3,4]
    
    def __getitem__(self, index):
        datum, label = self.load_datum(index)
        if self.transforms is not None:
            datum = self.transforms[self.split](datum)
        return {'data' : datum, 'label' : label}
    
    def parse_label(self, sample):
        if self.num_label == 1:
            self.idx2label = ['Normal', 'Wheezes', 'Crackles', 'Crackles&Wheezes']
            self.label2idx = {k: v for v,k in enumerate(self.idx2label)}
            label = torch.ones((1)) * self.label2idx[sample]
        elif self.num_label == 2:
            label = torch.zeros((2))
            self.idx2label = ['Wheezes', 'Crackles']
            self.label2idx = {k: v for v,k in enumerate(self.idx2label)}
            if sample == 'Normal':
                pass
            elif sample == 'Crackles&Wheezes':
                label = torch.ones((2))
            else:
                label[self.label2idx[sample]] = 1
        elif self.num_label == 3:
            label = torch.zeros((3))
            self.idx2label = ['Normal', 'Wheezes', 'Crackles']
            self.label2idx = {k: v for v,k in enumerate(self.idx2label)}
            if sample == 'Crackles&Wheezes':
                label[1:3] = 1
            else:
                label[self.label2idx[sample]] = 1
        elif self.num_label == 4:
            label = torch.zeros((4))
            self.idx2label = ['Normal', 'Wheezes', 'Crackles', 'Crackles&Wheezes']
            self.label2idx = {k: v for v,k in enumerate(self.idx2label)}
            label[self.label2idx[sample]] = 1
        return label
            
        
        