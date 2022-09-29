from dataman.base import base_dataset

import torch
import torchaudio
import torchvision

import os
import random
import numpy as np
from tqdm import tqdm

class icbhi_dataset(base_dataset):
    def __init__(self, configs, split='train'):
        super().__init__(configs, split)
        
        self.paths=os.listdir(configs.data_dir)
        
        if 'mixup' in configs.__dict__.keys():
            self.mixup = configs.mixup.do
            self.mixup_rate = configs.mixup.rate
        else:
            self.mixup = False
            self.mixup_rate = 0.
        self.fixed_length = configs.output_length
        dummy = torch.stft(torch.randn(1,self.sample_rate), n_fft = int(1e-3*self.window_size*self.sample_rate+1), 
                           hop_length=int(1e-3*self.hop_length*self.sample_rate),
                           window = torch.hann_window(int(1e-3*self.window_size*self.sample_rate+1))
                           )
        self.num_mel = dummy.shape[1]

        melscale = torchaudio.transforms.MelScale(sample_rate=16000, n_mels=128, n_stft=self.num_mel)
        
        self.fm = torchaudio.transforms.FrequencyMasking(int(0.2*self.num_mel))
        self.tm = torchaudio.transforms.TimeMasking(int(0.2*self.fixed_length))
        
        self.transforms = torchvision.transforms.Compose([
            torchvision.transforms.RandomCrop((self.num_mel, self.fixed_length)),
            self.fm,
            self.tm
            ])
        
    def to_multi_hot(self, ann:str):
        label = [0.]*len(ann)
        for i, an in enumerate(ann):
            if an == '1':
                label[i] = 1.0
        return label

    def to_one_hot(self, ann:str):
        label = [0]*(2**len(ann))
        label[int(ann,2)] = 1.0
        return label

    def to_int(self, ann):
        label = int(ann, 2)
        return label
    
    def __getitem__(self, index):
        datum = self.data[index]
        if self.mixup and random.random() < self.mixup_rate and self.split == 'train':
            mix_sample_idx = random.randint(0, len(self.data)-1)
            mix_datum = self.data[mix_sample_idx]

            mag, phase, mix_lambda = self._wav2fbank(datum, mix_datum)

            label1 = torch.from_numpy(np.array(self.labels[index])).unsqueeze(0)
            label2 = torch.from_numpy(np.array(self.labels[mix_sample_idx])).unsqueeze(0)

            label_indices = torch.cat((label1*mix_lambda, label2*(1-mix_lambda)), dim=0)
        else:
            mag, phase, mix_lambda = self._wav2fbank(datum)
            label = torch.from_numpy(np.array(self.labels[index])).unsqueeze(0)
            label_indices = torch.cat((label*(1-mix_lambda), label*mix_lambda), dim=0)

        if mag.shape[-1] < self.fixed_length:
            mag = mag.repeat(1, 1, self.fixed_length//mag.shape[-1] + 1)
            phase = phase.repeat(1, 1, self.fixed_length//phase.shape[-1] + 1)

        if self.split == 'train':
            magphase = self.transforms(torch.cat((mag.unsqueeze(0), phase.unsqueeze(0)), dim=0))
            mag = magphase[0]
            phase = magphase[1]
            mag = self.tm(mag)
            mag = self.fm(mag)
        else:
            mag = mag[:,:,:self.fixed_length]
            phase = phase[:,:,:self.fixed_length]

        return mag, phase, label_indices


    def initialize(self, paths, multi_label):
        wavs = [torch.empty(1)]*len(paths)
        labels = [np.empty(1)]*len(paths)
        for i, s in tqdm(enumerate(paths),total=len(paths)):
            sp = self.data_dir+"/"+s

            ann = s.split('_')[-1].split('.')[0]
            wavs[i] = sp
            if multi_label:
                ann = self.to_multi_hot(ann)
            else:
                # ann = self.to_int(ann, 2)
                # ann = self.to_multi_hot(ann)
                ann = self.to_one_hot(ann)
            labels[i] = ann
        self.data = wavs
        self.labels = labels
        return 0, 1