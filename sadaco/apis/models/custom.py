from utils.config_parser import parse_config_obj
import torch
import torch.nn as nn

def custom_model(yml_path):
    configs = parse_config_obj(yml_path)
    model = nn.Sequential()
    for k in configs.layers:
        layer = getattr(nn, configs.layers[k].name)(**configs.layers[k].params)
        model.add_module(str(k), layer)
    return model

if __name__ == "__main__":
    model = custom_model('custom_example.yml')
    dummy = torch.randn((1,3,224,224))
    print(dummy)
    out = model(dummy)
    print(out)