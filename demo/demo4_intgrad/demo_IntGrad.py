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
        IG = torch.zeros_like(inputs)
        baseline = torch.zeros_like(inputs)
        steps=master_cfg.explainer.ig.ig_steps
        for i in range(steps+1):
            scaled_inputs = torch.nn.Parameter(baseline + (inputs - baseline) * float(i) / steps)
            outputs = model(scaled_inputs.unsqueeze(0))
            model.zero_grad()
            loss_grads = torch.autograd.grad(outputs[0,k], scaled_inputs)
            IG += loss_grads[0]/steps
        IG = (inputs - baseline) * IG
        IG = torch.relu(IG)
        IG -= torch.min(torch.nn.Flatten()(IG).unsqueeze(1), dim=-1)[0][:,:,None]
        IG /= torch.clamp(torch.max(torch.nn.Flatten()(IG).unsqueeze(1), dim=-1)[0][:,:,None],
                min=1e-16)
        IG = IG.detach().cpu().numpy()
        if k == 0:
            plt.figure(figsize=(10, 4))
            display.specshow(min_max_scale(np.clip(min_max_scale(inputs.cpu().numpy()[0], 0, 1), 0, 0.2),0,1), y_axis='mel', sr=16000, hop_length=int(16*70), x_axis='time')
            plt.colorbar(format='%+2.0f dB')
            plt.tight_layout()
            plt.savefig(f'Input.png')
            plt.close()
        else:
            pass
        plt.figure(figsize=(10, 4))
        display.specshow(IG[0], y_axis='mel', sr=16000, hop_length=int(16*70), x_axis='time')
        plt.colorbar(format='%+2.0f dB')
        plt.tight_layout()
        plt.savefig(f'IntGrad-{clss[k]}.png')
        plt.close()