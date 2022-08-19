import torch
import torchaudio
from typing import DefaultDict, List

class Preprocessor:
    def __init__(self, preproc_modules : List =None):
        if preproc_modules is None:
            self.preproc_modules = []
        else:
            self.preproc_modules = preproc_modules
    def __call__(self, inputs:DefaultDict):
        for pm in self.preproc_modules:
            inputs = pm(inputs)
        return inputs
    def add_module(self, module):
        self.preproc_modules.append(module)
        
    def to(self, device):
        _ = [pm.to(device) for pm in self.preproc_modules]

class stft2meldb:
    def __init__(self, n_stft, n_mels=128, sample_rate=16000):
        self.n_stft=n_stft
        self.n_mels=n_mels
        self.melscale = torchaudio.transforms.MelScale(sample_rate=sample_rate, n_mels=n_mels, n_stft=n_stft)
    def __call__(self, inputs:DefaultDict):
        inputs['input'] = self.melscale(inputs['input'])
        inputs['input'] = torchaudio.functional.amplitude_to_DB(inputs['input'] , multiplier = 10., amin=1e-3, db_multiplier=1)
        return inputs
    def to(self, device):
        self.melscale = self.melscale.to(device)
    
class normalize:
    def __init__(self, mean=0, std=1):
        self.mean = mean
        self.std = std
    def __call__(self, inputs:DefaultDict):
        inputs['input'] = (inputs['input']  - self.mean) / self.std
        return inputs
    def to(self, device):
        pass
    
if __name__ == "__main__":
    my_preproc = Preprocessor()
    my_preproc.add_module(stft2meldb())
    my_preproc.add_module(normalize())
    dummy = torch.randn((1,1,524,128))
    print(my_preproc({'input':dummy, '1':None, '2':None}))