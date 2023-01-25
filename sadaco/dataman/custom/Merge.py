from ..icbhi.icbhi import ICBHI_Dataset
from ..HFLungV1.hflungv1 import HFLungV1_Dataset
from ..SPRSound.sprsound import SPRSound_Dataset

from typing import List
import torch
import torchaudio
import torchvision
from torch.utils.data import Dataset
from torch.utils.data._utils.collate import default_collate
import numpy as np

import os
import json 
import random

from ..frontend import spectrogram, fbank
'''
Merge ICBHI_Dataset, HF_LungV1_Dataset, SPRSound_Dataset into one dataset.
Since they have the same format and similar labels, we can merge them into one dataset.
Labels are:
    0: Normal
    1: Crackles
    2: Wheezes
    3: Crackles & Wheezes 
    4: Rhonchi
    5: Stridor
'''

class MergeDataset(Dataset):
    def __init__(self, configs, split='train'):
        super().__init__()
        self.num_label = configs.num_label
        self._parse_label = {}
        self._parse_label['icbhi'] = ICBHI_Dataset(configs,split,no_init=True).parse_label
        self._parse_label['sprsound'] = SPRSound_Dataset(configs,split,no_init=True).parse_label
        self._parse_label['hflungv1'] = HFLungV1_Dataset(configs,split,no_init=True).parse_label
        
        self.configs = configs
        if split in ['val', 'test', 'valid']:
            split = 'test'
        self.split= split   
        self.sample_rate = configs.sample_rate
        if configs.size_mode == 'size':
            self.window_size = configs.window_size
            self.hop_length = configs.hop_length
        elif configs.size_mode == 'time':
            # window & hop size in time(ms)
            self.window_size = int(1e-3*configs.window_size*self.sample_rate+1)
            self.hop_length = int(1e-3*configs.hop_length*self.sample_rate)
        
        self.num_mel = configs.num_mel
        self.fixed_length = configs.fixed_length
        
        datasets = self.configs.datasets
        self._data = {}
        self._labels = {}
        for dataset in datasets:
            root_dir = self.configs.__dict__[dataset]['data_dir']
            metadata = json.load(open(os.path.join(root_dir, 'meta.json')))
            
            data = metadata[self.split]['data']
            labels = metadata[self.split]['labels']
            self._data[dataset] = [os.path.join(root_dir, d) for d in data]
            self._labels[dataset]= [self._parse_label[dataset](l, self.num_label, multi_label=self.configs.multi_label) for l in labels]
        self.collate_fn = default_collate
        
        if self.split == 'train':
            self.data = []
            self.labels = []
            for dataset in datasets:
                self.data += self._data[dataset]
                self.labels += self._labels[dataset]
        elif self.split =='test':
            self.data = self._data[datasets[0]]
            self.labels = self._labels[datasets[0]]
        else:
            raise ValueError('Invalid split')
        
        if self.split == 'train':
            self.transforms = torchvision.transforms.Compose([
                torchvision.transforms.RandomCrop((self.num_mel, self.fixed_length), pad_if_needed=True),
            ])
                # torchaudio.transforms.FrequencyMasking(freq_mask_param=10),
                # torchaudio.transforms.TimeMasking(time_mask_param=10),])
        else:
            self.transforms = torchvision.transforms.Compose([
                torchvision.transforms.CenterCrop((self.num_mel, self.fixed_length))])
        
    def convert_wav(self, waveform, initialize=False):
        """Convert wav file to Mag+Phase matrix with STFT conversion.
        User can override this func to customize data format.

        :param waveform: Input wav file. Required shape : [Batch, Length]
        :type waveform: torch.Tensor
        :return: Tuple of mag, phase matrix
        :rtype: Tuple[Torch.Tensor]
        """        
        # 
        # !! Always return in tuple !!
        mel, (mag, phase) = fbank(waveform, num_mel_bins=self.num_mel, frame_length= self.window_size,
                                 frame_shift= self.hop_length, round_to_power_of_two= True, use_energy=False, htk_compat=True,
                                 sample_frequency = self.sample_rate, return_fft=True, size_mode='size', window_type='hanning',)
        # cart = torch.stft(waveform, n_fft = self.window_size, 
        #                    hop_length=self.hop_length,
        #                    window = torch.hann_window(self.window_size),
        #                    return_complex=True, pad_mode='reflect')
        # phase = torch.atan2(cart.imag, cart.real)
        # mag = cart.abs()**2
        mel = mel.permute(1,0).unsqueeze(0)

        if not initialize:
            mel = self.transforms(mel)
        return (mel)
    
    def recover_wav(self, mag, phase):
        """Inverse function of convert_wav. User should modify both of the functions when customizing.
        
        :param mag: Magnitude matrix from STFT.
        :type mag: torch.Tensor
        :param phase: Phase matrix from STFT
        :type phase: torch.Tensor
        :return: 
        :rtype: _type_
        """
        # mag, phase --> (F, T)
        mag = torch.sqrt(torch.relu(mag))
        recombine_magnitude_phase = torch.complex(
            mag*torch.cos(phase), mag*torch.sin(phase))
        recombine_magnitude_phase = recombine_magnitude_phase.permute(1, 0)
        rf = torch.fft.irfft(recombine_magnitude_phase, n=self.window_size)
        recon = torch.zeros(1, mag.shape[-1]*self.hop_length)
        for k in range(rf.shape[0]):
            recon[:, k*self.hop_length:k*self.hop_length+self.window_size] += rf[k]
        return recon
        
    def load_datum(self, index, initialize=False):
        if self.split=='train' and self.configs.mixup.do and random.random() < self.configs.mixup.rate:
            mix_sample_idx = random.randint(0, len(self.data)-1)
            waveform1, sample_rate = self.load_wav_from_path(self.data[index])
            waveform1 = self.loudness_normalization(waveform1)
            waveform2, sample_rate = self.load_wav_from_path(self.data[mix_sample_idx])
            waveform2 = self.loudness_normalization(waveform2)
            label1 = self.parse_label(self.labels[index])
            label2 = self.parse_label(self.labels[mix_sample_idx])
            if waveform1.shape[1] != waveform2.shape[1]:
                if waveform1.shape[1] > waveform2.shape[1]:
                    temp_wav = waveform2.repeat(1, waveform1.shape[-1]//waveform2.shape[-1] + 1)
                    waveform2 = temp_wav[0, 0:waveform1.shape[-1]]
                else:
                    randidx = np.random.randint(low=0, high=waveform2.shape[1]-waveform1.shape[1], size=(1,))
                    waveform2 = waveform2[0, randidx[0]:randidx[0]+waveform1.shape[1]]

            mix_lambda = np.random.beta(10, 10)

            mix_waveform = mix_lambda * waveform1 + (1 - mix_lambda) * waveform2
            waveform = mix_waveform - mix_waveform.mean()
        else:
            waveform, sample_rate = self.load_wav_from_path(self.data[index])
            waveform = self.loudness_normalization(waveform)
            label = self.parse_label(self.labels[index])
            label1 = label 
            label2 = label
            mix_lambda = 0.5
            
        datum = self.convert_wav(waveform=waveform, initialize=initialize)
            
        return datum, label1, label2, mix_lambda
    
    def load_wav_from_path(self, path):
        waveform, sample_rate = torchaudio.load(path, normalize=False)
        waveform = waveform.float()
        waveform = torchaudio.functional.resample(waveform,
                                                        sample_rate, self.sample_rate,
                                                        lowpass_filter_width=16,
                                                        rolloff=0.85,
                                                        resampling_method="kaiser_window",
                                                        beta=8.555504641634386,
                                                    )
        if len(waveform) <= self.window_size:
            waveform = torch.nn.functional.pad(waveform, (0, self.window_size))
        waveform = waveform.type(torch.FloatTensor) / (torch.iinfo(torch.int16).max +1)
        waveform = waveform - waveform.mean()
        return waveform, self.sample_rate
    
    def parse_label(self, sample):
        return sample

    @staticmethod
    def db_to_linear(samples):
        return 10.0 ** (samples / 20.0)

    def loudness_normalization(self, samples: torch.Tensor,
                            target_db: float = 15.0,
                            max_gain_db: float = 30.0):
        std = torch.std(samples) + 1e-9
        max_lin = self.db_to_linear(max_gain_db)
        target_lin = self.db_to_linear(target_db)
        gain = torch.min(torch.tensor(max_lin), target_lin / std)
        return gain * samples

    def initialize(self):
        import math
        from tqdm import tqdm
        mean = 0.
        mean_sq = 0.
        pixel_count = 0
        pbar = tqdm(range(len(self.data)))
        for p in pbar:
            datum, label = self.load_datum(p, initialize=True)
            img, _, _ = datum
            h, w = img.shape
            mean += img.sum()
            mean_sq += (img ** 2).sum()
            pixel_count += h * w
            pbar.set_postfix({'mean': mean, 'mean_sq': mean_sq, 'pixels': pixel_count})
        mean = mean / pixel_count
        mean_sq = mean_sq / pixel_count
        std = math.sqrt(mean_sq - mean ** 2)
        return mean, std
        
        
    def __getitem__(self, index):
        datum, label1, label2, mix_lambda = self.load_datum(index)
        return {'input' : datum, 'label' : label1, 'label2': label2, 'lam': mix_lambda}

    def __len__(self):
        return len(self.data)