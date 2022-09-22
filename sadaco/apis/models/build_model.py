from sadaco.apis.models._ast import ASTModel
from sadaco.apis.models.resnet import ResidualNet
# from apis.models.vgg import VGG
from sadaco.apis.models.custom import custom_model
from .cnn_moe import *
from .vgg import VGGish

#TODO: make all models more 'controllable' via configs
def build_model(configs):
    if configs.name in globals().keys():
        model = globals()[configs.name](**configs.params)
    else:
        raise ValueError(f'{configs.name} is not in the available Models List!')
    return model

if __name__ == "__main__":
    breakpoint()