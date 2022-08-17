import torch
import torchaudio
from typing import List

class Preprocessor():
    def __init__(self, preproc_modules : List =None):
        if preproc_modules is None:
            self.preproc_modules = []
        else:
            self.preproc_modules = preproc_modules
    def __call__(self, *inputs):
        for pm in self.preproc_modules:
            inputs = pm(*inputs)
        return inputs
    def add_module(self, module):
        self.preproc_modules.append(module)

class stft2meldb():
    def __init__(self, sample_rate=16000, n_mels=128, n_stft=524):
        self.melscale = torchaudio.transforms.MelScale(sample_rate=sample_rate, n_mels=n_mels, n_stft=n_stft)
    def __call__(self, *inputs):
        mags = self.melscale(inputs[0])
        mags = torchaudio.functional.amplitude_to_DB(mags, multiplier = 10., amin=1e-8, db_multiplier=1)
        return (mags,*inputs[1:])
    
class normalize():
    def __init__(self, mean=0, std=1):
        self.mean = mean
        self.std = std
    def __call__(self, *inputs):
        scaled = (inputs[0] - self.mean) / self.std
        return (scaled,*inputs[1:])
    
if __name__ == "__main__":
    my_preproc = Preprocessor()
    my_preproc.add_module(stft2meldb())
    my_preproc.add_module(normalize())
    dummy = torch.randn((1,1,524,128))
    print(my_preproc(dummy, None, None))