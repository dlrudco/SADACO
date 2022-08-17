import cv2
import torch
import torchaudio
import numpy as np
from apis.explain.hookman import FGHandler
from utils.config_parser import parse_config_obj
from apis.models import build_model
from apis.explain.visualize import spec_display

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
    inputs = torchaudio.functional.amplitude_to_DB(inputs, multiplier = 10., amin=1e-8, db_multiplier=1)
    
    norm_mean = -4.2677393
    norm_std = 4.5689974
    inputs = (inputs - norm_mean) / (norm_std * 2)
    return inputs

class demo_helper:
    def __init__(self, master_cfg, model_cfg):
        self.master_cfg = master_cfg
        self.model_cfg = model_cfg
        model = build_model(model_cfg)
        model = model.cuda()
        model.eval()
        checkpoint = torch.load(model_cfg.model_checkpoint)

        try:
            model.load_state_dict(checkpoint['state_dict'])
        except RuntimeError:
            model = torch.nn.DataParallel(model)
            model.load_state_dict(checkpoint['state_dict'])
            model = model.module
        
        layers = ['layer4.2.conv3', 'layer3.2.conv3', 'layer2.2.conv3', 'layer1.2.conv3', 
                'layer4.1.conv3', 'layer3.1.conv3']
        handler = FGHandler(model, layers)
        model.handler = handler
        self.model = model
        
    def do_inference(self, input_path, return_raw=False):
        inputs = load_input(input_path)
        outputs = self.model(inputs.unsqueeze(0))
        outputs = torch.softmax(outputs, dim=1)[0]
        text = f"Normal {outputs[0]*100:.2f} %, Wheeze {outputs[1]*100:.2f} %\nCrackle {outputs[2]*100:.2f} %, Both {outputs[3]*100:.2f} %"
        if return_raw:
            return inputs, outputs
        else:
            return text
    
    def do_explanation(self, input_path, method, cls):
        if method == 0:
            inputs, outputs = self.do_inference(input_path, return_raw=True)
            outputs[cls].backward()
            
            feature = self.model.handler.get_all_features('upscale', 'upscale')
            gradient = self.model.handler.get_all_grads('upscale', 'upscale')
            
            weight = np.mean(gradient.cpu().numpy(), axis=(2, 3))
            cam = feature.detach().cpu().numpy() * weight[:, :, np.newaxis, np.newaxis]
            cam = np.sum(cam, axis=1)
            cam = np.maximum(cam, 0)
            cam -= np.min(cam, axis=(1,2), keepdims=True)
            cam /= np.max(cam, axis=(1,2), keepdims=True)
            
            cam = np.array([cv2.resize(c, tuple(inputs.shape[-2:])) for c in cam])
            # cam = (1000*cam).astype(np.int32)
            # hist, bins = np.histogram(cam.flatten(), 1001, [0, 1001])
            # cdf = hist.cumsum()
            # cdf_m = np.ma.masked_equal(cdf,0)
            # cdf_m = (cdf_m - cdf_m.min())*1000/(cdf_m.max()-cdf_m.min())
            # cdf = np.ma.filled(cdf_m,0)
            # cam2 = cdf[cam]
            cam2 = inputs.cpu().numpy() * cam
            arr = spec_display(cam2[0].astype(np.float32), return_array=True)
            return arr
        elif method == 1:
            inputs = load_input(input_path)
            IG = torch.zeros_like(inputs)
            baseline = torch.zeros_like(inputs)
            steps=self.master_cfg.explainer.ig.ig_steps
            for i in range(steps+1):
                scaled_inputs = torch.nn.Parameter(baseline + (inputs - baseline) * float(i) / steps)
                outputs = self.model(scaled_inputs.unsqueeze(0))
                self.model.zero_grad()
                loss_grads = torch.autograd.grad(outputs[0, cls], scaled_inputs)
                IG += loss_grads[0]/steps
            IG = (inputs - baseline) * IG
            IG = torch.relu(IG)
            IG -= torch.min(torch.nn.Flatten()(IG).unsqueeze(1), dim=-1)[0][:,:,None]
            IG /= torch.clamp(torch.max(torch.nn.Flatten()(IG).unsqueeze(1), dim=-1)[0][:,:,None],
                    min=1e-16)
            IG = IG.detach().cpu().numpy()
            # IG = (1000*IG).astype(np.int32)
            # hist, bins = np.histogram(IG.flatten(), 1001, [0, 1001])
            # cdf = hist.cumsum()
            # cdf_m = np.ma.masked_equal(cdf,0)
            # cdf_m = (cdf_m - cdf_m.min())*1000/(cdf_m.max()-cdf_m.min())
            # cdf = np.ma.filled(cdf_m,0)
            # IG2 = cdf[IG]
            IG2 = inputs.cpu().numpy() * IG
            arr = spec_display(IG2[0].astype(np.float32), return_array=True)
            return arr
        else:
            raise ValueError
        
        
        