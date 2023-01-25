from typing import List
import torch
import torchaudio
from torch.utils.data import Dataset
from torch.utils.data._utils.collate import default_collate
import numpy as np

import os
import json 

class BaseDataset(Dataset):
    """Dataset Template
    """    
    def __init__(self, configs, split='train'):
        super().__init__()
        self.configs = configs
        self.split= split        
        self.sample_rate = configs.sample_rate
        if configs.size_mode == 'size':
            self.window_size = configs.window_size
            self.hop_length = configs.hop_length
        elif configs.size_mode == 'time':
            # window & hop size in time(ms)
            self.window_size = int(1e-3*configs.window_size*self.sample_rate+1)
            self.hop_length = int(1e-3*configs.hop_length*self.sample_rate)
        
        self.root_dir = self.configs.__dict__['data_dir']
        
        self.metadata = json.load(open(os.path.join(self.root_dir, 'meta.json')))
        
        self.data = self.metadata['data']
        self.labels = self.metadata['labels']
        
        self.collate_fn = default_collate
        
    def convert_wav(self, waveform):
        """Convert wav file to Mag+Phase matrix with STFT conversion.
        User can override this func to customize data format.

        :param waveform: Input wav file. Required shape : [Batch, Length]
        :type waveform: torch.Tensor
        :return: Tuple of mag, phase matrix
        :rtype: Tuple[Torch.Tensor]
        """        
        # 
        # !! Always return in tuple !!
        cart = torch.stft(waveform, n_fft = self.window_size, 
                           hop_length=self.hop_length,
                           window = torch.hann_window(self.window_size),
                           return_complex=True, pad_mode='reflect')
        phase = torch.atan2(cart.imag, cart.real)
        mag = cart.abs()**2
        return (mag, phase)
    
    def recover_wav(self, mag, phase):
        """Inverse function of convert_wav. User should modify both of the functions when customizing.
        
        :param mag: Magnitude matrix from STFT.
        :type mag: torch.Tensor
        :param phase: Phase matrix from STFT
        :type phase: torch.Tensor
        :return: 
        :rtype: _type_
        """
        mag = torch.sqrt(torch.relu(mag))
        recombine_magnitude_phase = torch.cat(
            [(mag*torch.cos(phase)).unsqueeze(-1), (mag*torch.sin(phase)).unsqueeze(-1)], 
            dim=-1)
        recon = torch.istft(recombine_magnitude_phase, 
                            n_fft = self.window_size, 
                            hop_length=self.hop_length,
                            window = torch.hann_window(self.window_size))
        return recon
        
    def load_datum(self, index):
        waveform, sample_rate = self.load_wav_from_path(self.data[index])
        waveform = self.loudness_normalization(waveform)
        datum = self.convert_wav(waveform=waveform)
        label = self.parse_label(self.labels[index])
        return datum, label
    
    @staticmethod
    def load_wav_from_path(path):
        waveform, sample_rate = torchaudio.load(path, normalize=False)
        waveform = waveform.type(torch.FloatTensor) / (torch.iinfo(torch.int16).max +1)
        return waveform, sample_rate
    
    def parse_label(self):
        raise NotImplementedError

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

    def __getitem__(self, index):
        datum, label = self.load_datum(index)
        return {'data' : datum, 'label' : label}

    def __len__(self):
        return len(self.data)
