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

def spec_display(spec:torch.Tensor, mag2db = False, sharpen:float=None, save_path=None, sr=16000, hop_length=int(16*70), return_array=False,
                 normalize_outliers=False, y_axis='mel', font_size=22):
    plt.rc('font', size=font_size)
    plt.rc('axes', titlesize=font_size)
    plt.rc('axes', labelsize=font_size)
    plt.rc('xtick', labelsize=font_size)
    plt.rc('ytick', labelsize=font_size)
    plt.rc('legend', fontsize=font_size)
    plt.rc('figure', titlesize=font_size+5)
    if mag2db:
        p2d =  torchaudio.transforms.AmplitudeToDB(stype='magnitude', top_db = 80)
        spec = p2d(spec)
    
    if sharpen is not None:
        # assume spec is in shape of CHW or at least HW
        if len(spec.shape) < 4:
            spec = spec[(None,)*(4-len(spec.shape))]
        spec= enhance_sharpeness(spec, sharpen)
        
    spec = spec.squeeze().numpy()
    f = plt.figure(figsize=(20, 8))
    if normalize_outliers:
        spec = clip_outliers(spec)
    display.specshow(spec, y_axis=y_axis, sr=sr, hop_length=hop_length, x_axis='time')
    plt.xlabel('Time(s)')
    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
    if return_array:
        arr = figure_to_array(f)
        plt.close()
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

def clip_outliers(data, range=[0.05, 0.95]):
    #Following Box & Whiskers like manner of determining inliers.
    q1, q3  = np.quantile(data, range)
    iqr = np.abs(0.5 * (q3 - q1))
    imin, imax = q1-iqr, q3+iqr
    data[data>imax] = imax
    data[data<imin] = imin
    return data

def enhance_sharpeness(data, magnitude=2):
    sharpen = torch.Tensor(np.array(
        [[[
            [0,-0.25*(magnitude-1),0],
            [-0.25*(magnitude-1),magnitude,-0.25*(magnitude-1)],
            [0,-0.25*(magnitude-1),0]
        ]]]
    ))
    data = torch.nn.functional.conv2d(data, sharpen, padding=1)
    return data

