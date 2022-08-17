import torch
import torchaudio
from utils.config_parser import parse_config_obj
from apis.models import build_model
from apis.traintest.demo import *
master_config = '../demo_materials/demo_configs.yml'
model_config = '../demo_materials/demo_model.yml'
input_path = '../demo_materials/wheeze.wav'
#180_1b4_Al_mc_AKGC417L_2_11.wav
model_checkpoint = '../demo_materials/demo_ckp.pth'

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
    
    inputs = load_input(input_path)
    
    outputs = model(inputs.unsqueeze(0))
    outputs = torch.softmax(outputs/outputs.abs().max(), dim=1)[0]
    print("Results of Classifying wheeze.wav")
    print(f"\tPred : [ Normal {outputs[0]*100:.2f} %, Wheeze {outputs[1]*100:.2f} %, Crackle {outputs[2]*100:.2f} %, Both {outputs[3]*100:.2f} % ]")
          
    print("\tTrue : Wheeze")