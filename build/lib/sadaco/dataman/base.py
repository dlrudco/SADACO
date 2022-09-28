from typing import List
import torch
import torchaudio
from torch.utils.data import Dataset
import numpy as np

class base_dataset(Dataset):
    def __init__(self, configs, split='train'):
        super().__init__()
        self.split= split        
        self.sample_rate = configs.sample_rate
        if configs.size_mode == 'size':
            self.window_size = configs.window_size
            self.hop_length = configs.hop_length
        elif configs.size_mode == 'time':
            # window & hop size in time(ms)
            self.window_size = int(1e-3*configs.window_size*self.sample_rate+1)
            self.hop_length = int(1e-3*configs.hop_length*self.sample_rate)
        
    def convert_wav(self, waveform):
        # User can override this func to customize data format.
        # !! Always return in tuple !!
        cart = torch.stft(waveform, n_fft = self.window_size, 
                           hop_length=self.hop_length,
                           window = torch.hann_window(self.window_size)
                           )
        phase = torch.atan2(cart[:,:,:,1], cart[:,:,:,0])
        mag = cart[:,:,:,0]**2 + cart[...,1]**2
        return (mag, phase)
        
    def _wav2fbank(self, filename, filename2=None):
        if filename2 == None:
            waveform, sr = torchaudio.load(filename)
            waveform = waveform - waveform.mean()
        else:
            waveform1, sr = torchaudio.load(filename)
            waveform2, _ = torchaudio.load(filename2)

            waveform1 = waveform1 - waveform1.mean()
            waveform2 = waveform2 - waveform2.mean()

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

        wav2freq = self.convert_wav(waveform)
        
        if filename2 == None:
            return (*wav2freq, 0)
        else:
            return (*wav2freq, mix_lambda) 

    def __len__(self):
        return len(self.data)

    def recover_stft(self, mag, phase):
        # mag = torchaudio.functional.DB_to_amplitude(mag, power=1)
        # mag = mag * self.norm_std * 2 + self.norm_mean
        mag = torch.sqrt(torch.relu(mag))
        recombine_magnitude_phase = torch.cat(
            [(mag*torch.cos(phase)).unsqueeze(-1), (mag*torch.sin(phase)).unsqueeze(-1)], 
            dim=-1)
        recon = torch.istft(recombine_magnitude_phase, 
                            n_fft = int(1e-3*self.window_size*self.sample_rate+1), 
                            hop_length=int(1e-3*self.hop_length*self.sample_rate),
                            window = torch.hann_window(int(1e-3*self.window_size*self.sample_rate+1)))
        return recon
    
    def amp2db(self, tensor):
        tensor = torchaudio.functional.amplitude_to_DB(tensor, multiplier = 10., amin=1e-8, db_multiplier=1)
        return tensor