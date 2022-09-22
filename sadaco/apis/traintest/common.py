import torch
import torchaudio

def load_input(input_path, mode=0, n_mels=128, n_fft=70, hop_length=25, sample_rate=16000):
    waveform, _ = torchaudio.load(input_path)
    waveform = waveform - waveform.mean()
    
    cart = torch.stft(waveform, n_fft = int(1e-3*n_fft*sample_rate+1), 
                           hop_length=int(1e-3*hop_length*sample_rate),
                           window = torch.hann_window(int(1e-3*n_fft*sample_rate+1))
                           )
    phase = torch.atan2(cart[:,:,:,1], cart[:,:,:,0])
    mag = cart[:,:,:,0]**2 + cart[...,1]**2
    if mag.shape[-1] < 128:
        mag = mag.repeat(1, 1, 128//mag.shape[-1] + 1)
        mag = mag[:,:,:128]
    else:
        mag = mag[:,:,:128]
    if mode == 0:
        return (mag, phase)
    elif mode == 1 :
        melscale = torchaudio.transforms.MelScale(sample_rate=16000, n_mels=n_mels, n_stft=mag.shape[1]).cuda()
        inputs = melscale(mag.cuda().float())
        return inputs
    else:
        melscale = torchaudio.transforms.MelScale(sample_rate=16000, n_mels=n_mels, n_stft=mag.shape[1]).cuda()
        inputs = melscale(mag.cuda().float())
        inputs = torchaudio.functional.amplitude_to_DB(inputs, multiplier = 10., amin=1e-8, db_multiplier=1)
        return inputs