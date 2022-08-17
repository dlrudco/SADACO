import os
import numpy as np
from torch.utils.data import Dataset
import torchvision
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
import random
from tqdm import tqdm
import torch
import librosa
import torchaudio
torchaudio.set_audio_backend("soundfile")
class RespiDatasetSTFT(Dataset):
    def __init__(self, split, mixup, initialize=True, data_dir="dataset/spec_cut", multi_label=False, mean=None, std=None, 
                 fixed_length=None, sr=16000, num_mel=None, hop_length=5, window_size=70):
        super(RespiDatasetSTFT, self).__init__()
        self.split=split
        self.mixup=mixup
        self.hop_length = hop_length
        self.window_size = window_size
        assert self.split in ['train', 'val'], "split must be either train or val"
        if self.split == 'train':
            self.train_data = []
        else:
            self.val_data = []
        self.data_dir=data_dir
        self.path=os.listdir(self.data_dir)
        # only used if data need to be in fixed length
        self.fixed_length=fixed_length
        self.multi_label = multi_label
        self.weights = []
        if initialize:
            if mean is None or std is None:
                self.mean, self.std = self.initialize(self.path, self.multi_label)
            else:
                self.mean = mean
                self.std = std
            print(self.mean, self.std)
        else:
            self.mean = mean
            self.std = std
        self.sample_rate = sr
        dummy = torch.stft(torch.randn(1,self.sample_rate), n_fft = int(1e-3*self.window_size*self.sample_rate+1), 
                           hop_length=int(1e-3*self.hop_length*self.sample_rate),
                           window = torch.hann_window(int(1e-3*self.window_size*self.sample_rate+1))
                           )
        self.num_mel = dummy.shape[1]

        self.transforms = torchvision.transforms.Compose([
            torchvision.transforms.RandomCrop((self.num_mel, self.fixed_length))
            ])
        
        self.fm = torchaudio.transforms.FrequencyMasking(int(0.2*self.num_mel))
        self.tm = torchaudio.transforms.TimeMasking(int(0.2*self.fixed_length))
        self.norm_mean = -4.2677393
        self.norm_std = 4.5689974
        

    def _wav2fbank(self, filename, filename2=None):
        # mixup
        if filename2 == None:
            waveform, sr = torchaudio.load(filename)
            waveform = waveform - waveform.mean()
        # mixup
        else:
            waveform1, sr = torchaudio.load(filename)
            waveform2, _ = torchaudio.load(filename2)

            waveform1 = waveform1 - waveform1.mean()
            waveform2 = waveform2 - waveform2.mean()
            # breakpoint()
            if waveform1.shape[1] != waveform2.shape[1]:
                if waveform1.shape[1] > waveform2.shape[1]:
                    # padding
                    # temp_wav = torch.zeros(1, waveform1.shape[1])
                    # temp_wav[0, 0:waveform2.shape[1]] = waveform2
                    # waveform2 = temp_wav
                    # duplicating
                    temp_wav = waveform2.repeat(1, waveform1.shape[-1]//waveform2.shape[-1] + 1)
                    waveform2 = temp_wav[0, 0:waveform1.shape[-1]]
                else:
                    # front cutting
                    # waveform2 = waveform2[0, 0:waveform1.shape[1]]
                    # random cutting
                    randidx = np.random.randint(low=0, high=waveform2.shape[1]-waveform1.shape[1], size=(1,))
                    waveform2 = waveform2[0, randidx[0]:randidx[0]+waveform1.shape[1]]

            mix_lambda = np.random.beta(10, 10)

            mix_waveform = mix_lambda * waveform1 + (1 - mix_lambda) * waveform2
            waveform = mix_waveform - mix_waveform.mean()

        # fbank = torchaudio.compliance.kaldi.fbank(waveform, htk_compat=True, sample_frequency=sr, use_energy=False,
        #                                           window_type='hanning', num_mel_bins=self.num_mel, dither=0.0, frame_shift=10)

        # fbank=fbank.permute(1,0).unsqueeze(0)
        cart = torch.stft(waveform, n_fft = int(1e-3*self.window_size*self.sample_rate+1), 
                           hop_length=int(1e-3*self.hop_length*self.sample_rate),
                           window = torch.hann_window(int(1e-3*self.window_size*self.sample_rate+1))
                           )
        phase = torch.atan2(cart[:,:,:,1], cart[:,:,:,0])
        mag = cart[:,:,:,0]**2 + cart[...,1]**2
        
        if filename2 == None:
            return mag, phase, 0
        else:
            return mag, phase, mix_lambda

    def __getitem__(self, index):
        """
        returns: image, audio, nframes
        where image is a FloatTensor of size (3, H, W)
        audio is a FloatTensor of size (N_freq, N_frames) for spectrogram, or (N_frames) for waveform
        nframes is an integer
        """
        # do mix-up for this sample (controlled by the given mixup rate)
        if self.mixup and random.random() < 0.5 and self.split == 'train':
            # print('MIXUP')
            datum = self.train_data[index]
            mix_sample_idx = random.randint(0, len(self.train_data)-1)
            mix_datum = self.train_data[mix_sample_idx]

            mag, phase, mix_lambda = self._wav2fbank(datum, mix_datum)
            # initialize the label
            label1 = torch.from_numpy(np.array(self.labels[index])).unsqueeze(0)
            label2 = torch.from_numpy(np.array(self.labels[mix_sample_idx])).unsqueeze(0)
            # label_indices = label1 * mix_lambda + label2 * (1.0-mix_lambda)
            label_indices = torch.cat((label1*mix_lambda, label2*(1-mix_lambda)), dim=0)
            # print(label1, label2, mix_lambda, label_indices)
        # if not do mixup
        else:
            if self.split == 'train':
                datum = self.train_data[index]
            else:
                datum = self.val_data[index]
            mag, phase, mix_lambda = self._wav2fbank(datum)
            label = torch.from_numpy(np.array(self.labels[index])).unsqueeze(0)
            label_indices = torch.cat((label*(1-mix_lambda), label*mix_lambda), dim=0)
        # mag = torchaudio.functional.amplitude_to_DB(mag, multiplier = 10., amin=1e-8, db_multiplier=1)
        # normalize the input
        if mag.shape[-1] < self.fixed_length:
            mag = mag.repeat(1, 1, self.fixed_length//mag.shape[-1] + 1)
            phase = phase.repeat(1, 1, self.fixed_length//phase.shape[-1] + 1)
        # mag = (mag - self.norm_mean) / (self.norm_std * 2)
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
    
                ann = self.to_one_hot(ann)
            labels[i] = ann
        self.data = wavs
        self.labels = labels
        return 0, 1
    

    def to_multi_hot(self, ann):
        label = [0.]*len(ann)
        for i, an in enumerate(ann):
            if an == '1':
                label[i] = 1.0
        return label

    def to_one_hot(self, ann):
        label = [0]*(2**len(ann))
        label[int(ann,2)] = 1.0
        return label

    def to_int(self, ann):
        label = int(ann, 2)
        return label

    def __len__(self):
        return len(self.data)

    def recover(self, mag, phase):
        mag = torch.sqrt(torch.relu(mag))
        recombine_magnitude_phase = torch.cat(
            [(mag*torch.cos(phase)).unsqueeze(-1), (mag*torch.sin(phase)).unsqueeze(-1)], 
            dim=-1)
        recon = torch.istft(recombine_magnitude_phase, 
                            n_fft = int(1e-3*self.window_size*self.sample_rate+1), 
                            hop_length=int(1e-3*self.hop_length*self.sample_rate),
                            window = torch.hann_window(int(1e-3*self.window_size*self.sample_rate+1)))
        return recon

train_dataset = RespiDatasetSTFT(split='train', data_dir='/train', initialize=True, 
            num_mel=128, multi_label=False, fixed_length=128, mixup=args.mixup)