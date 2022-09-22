import cv2
import torch
import torchaudio
import numpy as np
from librosa import display
import matplotlib.pyplot as plt

from apis.explain.hookman import FGHandler
from utils.config_parser import parse_config_obj
from apis.models import build_model
from apis.traintest.demo import *
from apis.explain.visualize import spec_display


master_config = '../demo_materials/demo_configs.yml'
model_config = '../demo_materials/demo_model.yml'
input_path = '../demo_materials/wheeze.wav'
#180_1b4_Al_mc_AKGC417L_2_11.wav
model_checkpoint = '../demo_materials/demo_ckp.pth'

def min_max_scale(samples, min, max):
    samples = samples-samples.min()
    samples = samples / samples.max()
    samples = (max - min) * samples + min
    return samples

if __name__ == "__main__":
    master_cfg = parse_config_obj(master_config)
    model_cfg = parse_config_obj(model_config)
    
    model = build_model(model_cfg)
    model = model.cuda()
    model.eval()
    checkpoint = torch.load(model_checkpoint)

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
    
    inputs = load_input(input_path)
    
    outputs = model(inputs.unsqueeze(0))
    outputs = torch.softmax(outputs/outputs.abs().max(), dim=1)[0]
    print("Results of Classifying wheeze.wav")
    print(f"\tPred : [ Normal {outputs[0]*100:.2f} %, Wheeze {outputs[1]*100:.2f} %, Crackle {outputs[2]*100:.2f} %, Both {outputs[3]*100:.2f} % ]")
        
    print("\tTrue : Wheeze")
    clss = ['Normal', 'Wheeze', 'Crackle', 'Both']
    for k in range(4):
        model.zero_grad()
        outputs = model(inputs.unsqueeze(0))
        
        outputs = torch.softmax(outputs, dim=1)[0]
        outputs[k].backward()
        
        feature = handler.get_all_features('upscale', 'upscale')
        gradient = handler.get_all_grads('upscale', 'upscale')
        
        weight = np.mean(gradient.cpu().numpy(), axis=(2, 3))
        cam = feature.detach().cpu().numpy() * weight[:, :, np.newaxis, np.newaxis]
        cam = np.sum(cam, axis=1)
        cam = np.maximum(cam, 0)
        cam -= np.min(cam, axis=(1,2), keepdims=True)
        cam /= np.max(cam, axis=(1,2), keepdims=True)
        
        cam = np.array([cv2.resize(c, tuple(inputs.shape[-2:])) for c in cam])

        if k == 0:
            spec_display(min_max_scale(inputs.cpu().numpy()[0], 0, 1), f'Input.png')
        else:
            pass
        spec_display(cam[0], f'GradCAM-{clss[k]}.png')