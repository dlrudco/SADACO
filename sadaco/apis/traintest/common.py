import torch
import torchaudio

def load_input(input_path, mode='stft', window_size=70, hop_length=25, sample_rate=16000):
    waveform, sr = torchaudio.load(input_path)
    if sr != sample_rate:
        print(f"Warning!! : File sample rate {sr}kHz != option sample rate {sample_rate}")
    waveform = waveform - waveform.mean()
    if mode == 'waveform':
        return waveform, sr
    
    cart = torch.stft(waveform, n_fft = int(1e-3*window_size*sample_rate+1), 
                           hop_length=int(1e-3*hop_length*sample_rate),
                           window = torch.hann_window(int(1e-3*window_size*sample_rate+1)),
                           return_complex=True, pad_mode='reflect')
    phase = torch.atan2(cart.imag, cart.real)
    mag = cart.abs()
    
    return mag, phase
    
def load_wav(input_path):
    waveform, sr = torchaudio.load(input_path)
    return waveform, sr
    
def stft2mel(mag, n_mels=128, sample_rate=16000):
    norm_mean = -4.2677393
    norm_std = 4.5689974
    melscale = torchaudio.transforms.MelScale(sample_rate=sample_rate, n_mels=n_mels, n_stft=mag.shape[-2]).to(mag.device)
    # p2d =  torchaudio.transforms.AmplitudeToDB(stype='magnitude', top_db = 80)
    inputs = torchaudio.functional.amplitude_to_DB(melscale(mag), multiplier = 10., amin=1e-8, db_multiplier=1)
    # inputs = p2d(melscale(mag))
    inputs = (inputs - norm_mean) / (norm_std**2)
    return inputs
    
def recover_wav(mag, phase, window_size, hop_length, sample_rate):
    mag = torch.relu(mag)
    recombine_magnitude_phase = torch.cat(
        [(mag*torch.cos(phase)).unsqueeze(-1), (mag*torch.sin(phase)).unsqueeze(-1)], 
        dim=-1)
    recon = torch.istft(recombine_magnitude_phase, 
                        n_fft = int(1e-3*window_size*sample_rate+1), 
                        hop_length=int(1e-3*hop_length*sample_rate),
                        window = torch.hann_window(int(1e-3*window_size*sample_rate+1)))
    return recon