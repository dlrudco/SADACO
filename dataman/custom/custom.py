from dataman.base import base_dataset
import torch
import random
import numpy as np

class custom_dataset(base_dataset):
    def __init__(self, configs, split='train'):
        super().__init__(configs, split)
    
    def __getitem__(self, index):
        raise NotImplementedError

    def initialize(self, paths, multi_label):
        raise NotImplementedError