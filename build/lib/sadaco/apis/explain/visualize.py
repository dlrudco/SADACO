from librosa import display
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchaudio

def min_max_scale(samples, min, max):
    samples = samples-samples.min()
    samples = samples / samples.max()
    samples = (max - min) * samples + min
    return samples

def load_input(input_path):
    waveform, _ = torchaudio.load(input_path)
    waveform = waveform - waveform.mean()
    
    cart = torch.stft(waveform, n_fft = int(1e-3*70*16000+1), 
                           hop_length=int(1e-3*25*16000),
                           window = torch.hann_window(int(1e-3*70*16000+1))
                           )
    phase = torch.atan2(cart[:,:,:,1], cart[:,:,:,0])
    mag = cart[:,:,:,0]**2 + cart[...,1]**2
    if mag.shape[-1] < 128:
        mag = mag.repeat(1, 1, 128//mag.shape[-1] + 1)
        mag = mag[:,:,:128]
    else:
        mag = mag[:,:,:128]
    melscale = torchaudio.transforms.MelScale(sample_rate=16000, n_mels=128, n_stft=mag.shape[1]).cuda()
    inputs = melscale(mag.cuda().float())
    
    norm_mean = -4.2677393
    norm_std = 4.5689974
    inputs = (inputs - norm_mean) / (norm_std * 2)
    return inputs

def figure_to_array(fig):
    fig.canvas.draw()
    return np.array(fig.canvas.renderer._renderer)

def spec_display(spec, save_path=None, sr=16000, hop_length=int(16*70), return_array=False):
    f = plt.figure(figsize=(4, 4))
    display.specshow(spec, y_axis='mel', sr=sr, hop_length=hop_length, x_axis='time')
    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
    if return_array:
        arr = figure_to_array(f)
        return arr
    plt.close()
    
def get_input_img(sample_path):
    spec = load_input(sample_path)
    spec = min_max_scale(spec.cpu().numpy()[0],0,1)
    spec = (1000*spec).astype(np.int32)
    hist, bins = np.histogram(spec.flatten(), 1001, [0, 1001])
    cdf = hist.cumsum()
    cdf_m = np.ma.masked_equal(cdf,0)
    cdf_m = (cdf_m - cdf_m.min())*1000/(cdf_m.max()-cdf_m.min())
    cdf = np.ma.filled(cdf_m,0)
    spec2 = cdf[spec]
    arr = spec_display(spec2.astype(np.float32)/1000, return_array=True)
    return arr