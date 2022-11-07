import torch
import torchaudio
from .hookman import FGHandler
import numpy as np
import cv2

class BaseExplainer(FGHandler): 
    def __init__(self, model, layers=None):
        super().__init__(model, layer_name = layers)
        
class GradcamExplainer(BaseExplainer):
    def __init__(self, model, layers):
        super().__init__(model, layers)
        
    def forward(self, inputs):
        pass
    
def apply_mask(data, mask):
    return data*mask

def TransformGrid(mask, n_fft, n_mels, sample_rate, mode = 'mel2stft', return_bool=True):
    # only consider transforming mask 'coverage'
    mel_filters = torchaudio.functional.melscale_fbanks(
        int(n_fft // 2 + 1),
        n_mels=n_mels,
        f_min=0.0,
        f_max=sample_rate / 2.0,
        sample_rate=sample_rate,
        norm="slaney",
    )
    mask = mask.transpose(-2, -1)
    # mel_filters are in [n_stft, n_mels]
    if mode == 'mel2stft':
        #transform [..., n_mels, n_frames] into [..., n_stft, n_frames]
        if return_bool:
            mask = torch.matmul(mask,mel_filters.t())
        else:
            mask = torch.matmul(mask,torch.linalg.pinv(mel_filters))
    elif mode == 'stft2mel':
        mask = torch.matmul(mask, mel_filters)
        
    if return_bool:
        mask = mask.bool()
        
    return mask.transpose(-2, -1)

def demo_explanation(model, data, method, cls, preprocessings=None, postprocessings=None):
    
    inputs = data
    
    if preprocessings is not None:
        for p in preprocessings:
            inputs = p(inputs)        
        
    if method == 0:
        model.zero_grad()
        outputs = model(inputs)
        outputs[0][cls].backward()
        
        feature = model.handler.get_all_features('upscale', 'upscale')
        gradient = model.handler.get_all_grads('upscale', 'upscale')
        
        weight = np.mean(gradient.cpu().numpy(), axis=(2, 3))
        cam = feature.detach().cpu().numpy() * weight[:, :, np.newaxis, np.newaxis]
        cam = np.sum(cam, axis=1)
        cam = np.maximum(cam, 0)
        cam -= np.min(cam, axis=(1,2), keepdims=True)
        cam /= np.max(cam, axis=(1,2), keepdims=True)
        
        cam = torch.Tensor(np.array([cv2.resize(c, tuple(inputs.shape[-2:][::-1])) for c in cam]))
        if postprocessings is not None:
            for pp in postprocessings:
                cam = pp(cam)
        overlay = data[0].cpu() * cam
        
        return cam, overlay
    elif method == 1:
        model.zero_grad()
        IG = torch.zeros_like(data)
        baseline = torch.zeros_like(data)
        steps = 20
        for i in range(steps+1):
            inputs = torch.nn.Parameter(baseline + (data - baseline) * float(i) / steps)
            scaled_inputs = inputs
            if preprocessings is not None:
                for p in preprocessings:
                    scaled_inputs = p(scaled_inputs)
            outputs = model(scaled_inputs)
            model.zero_grad()
            loss_grads = torch.autograd.grad(outputs[0, cls], inputs)
            IG += loss_grads[0]/steps
        IG = (data - baseline) * IG
        IG = torch.relu(IG)
        IG -= torch.min(torch.nn.Flatten()(IG).unsqueeze(1), dim=-1)[0][:,:,None]
        IG /= torch.clamp(torch.max(torch.nn.Flatten()(IG).unsqueeze(1), dim=-1)[0][:,:,None],
                min=1e-16)
        IG = IG.detach().cpu()
        overlay = data[0].cpu() * IG
        return IG, overlay
    else:
        raise ValueError