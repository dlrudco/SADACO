from typing import Dict
from sadaco.dataman.base import BaseDataset

import torch
import torchaudio
import torchvision

import os
import random
import numpy as np
from tqdm import tqdm

class HFLungV1_Dataset(BaseDataset):
    def __init__(self, configs, split='train', transforms:Dict=None, no_init=False):
        if no_init:
            pass
        else:
            super().__init__(configs, split)
            
            if 'mixup' in configs.__dict__.keys():
                self.mixup = configs.mixup.do
                self.mixup_rate = configs.mixup.rate
            else:
                self.mixup = False
                self.mixup_rate = 0.
            self.fixed_length = configs.fixed_length
            self.transforms = transforms
            
            if configs.num_label is not None:
                self.num_label = int(configs.num_label)
            else:
                self.num_label = None
    
    def __getitem__(self, index):
        datum, label = self.load_datum(index)
        if self.transforms is not None:
            datum = self.transforms[self.split](datum)
        return {'data' : datum, 'label' : label}
    
    def parse_label(self, sample, num_label=None, multi_label=False):
        if num_label is None:
            num_label = self.num_label
            
        if multi_label:
            if num_label == 4:
                label = torch.zeros((4))
                self.idx2label = ['Wheezes', 'Crackles', 'Rhonchi','Stridor']
                self.label2idx = {k: v for v,k in enumerate(self.idx2label)}
                if sample == 'Crackles&Wheezes':
                    label[0:2] = 1
                elif sample == ['Normal']:
                    pass
                else:
                    for s in sample:
                        label[self.label2idx[s]] = 1
            else:
                raise ValueError('num_label should be 1, 5, 11 or 4')
        else:
            if 'Crackles' in sample and 'Wheezes' in sample:
                sample = 'Crackles&Wheezes'
            elif 'Crackles' in sample and 'Stridor' in sample:
                sample = 'Crackles&Stridor'
            elif 'Rhonchi' in sample and 'Wheezes' in sample:
                sample = 'Rhonchi&Wheezes'
            elif 'Rhonchi' in sample and 'Stridor' in sample:
                sample = 'Rhonchi&Stridor'
            elif 'Stridor' in sample and 'Wheezes' in sample:
                sample = 'Stridor&Wheezes'
            elif 'Crackles' in sample and 'Rhonchi' in sample:
                sample = 'Crackles&Rhonchi'
            else:
                assert len(list(set(sample))) == 1
                sample = list(set(sample))[0]
            if num_label == 1:
                self.idx2label = ['Normal', 'Wheezes', 'Crackles', 'Rhonchi','Stridor']
                self.label2idx = {k: v for v,k in enumerate(self.idx2label)}
                label = torch.ones((1)) * self.label2idx[sample]
            elif num_label == 5:
                label = torch.zeros((5))
                self.idx2label = ['Normal', 'Wheezes', 'Crackles', 'Rhonchi', 'Stridor']
                self.label2idx = {k: v for v,k in enumerate(self.idx2label)}
                label[self.label2idx[sample]] = 1
            elif num_label == 11:
                label = torch.zeros((11))
                self.idx2label = ['Normal', 'Wheezes', 'Crackles', 'Crackles&Wheezes', 'Rhonchi','Stridor',
                                'Crackles&Stridor', 'Rhonchi&Wheezes', 'Stridor&Wheezes', 'Crackles&Rhonchi', 'Rhonchi&Stridor']
                self.label2idx = {k: v for v,k in enumerate(self.idx2label)}             
                label[self.label2idx[sample]] = 1
            else:
                raise ValueError('num_label should be 1, 5, 11 or 4')
        return label
            
        
        