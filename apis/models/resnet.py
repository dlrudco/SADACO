from re import L
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn import init
from apis.models.cbam import *


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, use_cbam=False):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=False)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.use_cbam = use_cbam
        if use_cbam:
            self.cbam = CBAM( planes, 16 )


    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)


        if self.downsample is not None:
            residual = self.downsample(x)

        if self.use_cbam:
            out = self.cbam(out)
            # out = out + cbam_out

        out += residual
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, use_cbam=False):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=False)
        self.downsample = downsample
        self.stride = stride

        if use_cbam:
            self.cbam = CBAM( planes * 4, 16 )
        else:
            self.cbam = None

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        if not self.cbam is None:
            out += self.cbam(out)

        out += residual
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    def __init__(self, block, layers,  network_type, num_classes, att_type, in_channel):
        
        super(ResNet, self).__init__()
        self.network_type = network_type
        # different model config between ImageNet and CIFAR 
        if network_type == "ImageNet":
            self.inplanes = 64
            self.conv1 = nn.Conv2d(in_channel, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            # self.avgpool = nn.AvgPool2d(7)
            self.avgpool = nn.AdaptiveAvgPool2d(1)
            self.bn1 = nn.BatchNorm2d(64)
        else:
            self.inplanes = 16
            self.conv1 = nn.Conv2d(in_channel, 16, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(16)

        
        self.relu = nn.ReLU(inplace=False)

        if self.network_type == "ImageNet":
            self.layer1 = self._make_layer(block, 64,  layers[0], att_type=att_type)
            self.layer2 = self._make_layer(block, 128, layers[1], stride=2, att_type=att_type)
            self.layer3 = self._make_layer(block, 256, layers[2], stride=2, att_type=att_type)
            self.layer4 = self._make_layer(block, 512, layers[3], stride=2, att_type=att_type)

            self.fc = nn.Linear(512 * block.expansion, num_classes)
        elif self.network_type == "Sound":
            self.layer1 = self._make_layer(block, 16,  layers[0], stride=2, att_type=att_type)
            self.layer2 = self._make_layer(block, 24, layers[1], stride=2, att_type=att_type)
            self.layer3 = self._make_layer(block, 32, layers[2], stride=2, att_type=att_type)
            self.layer4 = self._make_layer(block, 64, layers[3], stride=1, att_type=att_type)

            self.fc = nn.Linear(512 * block.expansion, num_classes)
        else:
            self.layer1 = self._make_layer(block, 16,  layers[0], att_type=att_type)
            self.layer2 = self._make_layer(block, 32, layers[1], stride=2, att_type=att_type)
            self.layer3 = self._make_layer(block, 64, layers[2], stride=2, att_type=att_type)

            self.fc = nn.Linear(64 * block.expansion, num_classes)

        init.kaiming_normal(self.fc.weight)
        for key in self.state_dict():
            if key.split('.')[-1]=="weight":
                if "conv" in key:
                    init.kaiming_normal(self.state_dict()[key], mode='fan_out')
                if "bn" in key:
                    if "SpatialGate" in key:
                        self.state_dict()[key][...] = 0
                    else:
                        self.state_dict()[key][...] = 1
            elif key.split(".")[-1]=='bias':
                self.state_dict()[key][...] = 0

        if att_type=='CBAM':
            self.CBAM_params = []
            self.W1_params = []
            for name, param in self.state_dict().items():
                if 'cbam' in name:
                    self.CBAM_params.append({'params': param})
                else:
                    self.W1_params.append({'params':param})


    def _make_layer(self, block, planes, blocks, stride=1, att_type=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, use_cbam=att_type=='CBAM'))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, use_cbam=att_type=='CBAM'))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if self.network_type == "ImageNet":
            x = self.maxpool(x)

        x = self.layer1(x)

        x = self.layer2(x)

        x = self.layer3(x)


        if self.network_type == "ImageNet":
            x = self.layer4(x)
            x = self.avgpool(x)
        elif self.network_type == "Sound":
            x = self.layer4(x)
            x = F.avg_pool2d(x, 8)
        else:
            x = F.avg_pool2d(x, 8)
        x = x.view(x.size(0), -1)
        # breakpoint()
        x = self.fc(x)
        return x

def ResidualNet(network_type, depth, num_classes, att_type, in_channel=3, pretrained=False):

    assert network_type in ["Sound", "ImageNet", "CIFAR10", "CIFAR100"], "network type should be ImageNet or CIFAR10 / CIFAR100"
    assert depth in [8, 18, 34, 50, 101], 'network depth should be 18, 34, 50 or 101'
    if network_type == "Sound":
        network_type = "ImageNet"
    else:
        pass
    if depth == 8:
        model = ResNet(BasicBlock, [1, 1, 1], network_type, num_classes, att_type, in_channel)
    elif depth == 18:
        model = ResNet(BasicBlock, [2, 2, 2, 2], network_type, num_classes, att_type, in_channel)
        if pretrained:
            import torchvision.models as models
            pt_model = models.resnet18(pretrained=True)
        else:
            pass
    elif depth == 34:
        model = ResNet(BasicBlock, [3, 4, 6, 3], network_type, num_classes, att_type, in_channel)
        if pretrained:
            import torchvision.models as models
            pt_model = models.resnet34(pretrained=True)
        else:
            pass
    elif depth == 50:
        model = ResNet(Bottleneck, [3, 4, 6, 3], network_type, num_classes, att_type, in_channel)
        if pretrained:
            import torchvision.models as models
            pt_model = models.resnet50(pretrained=True)
        else:
            pass
    elif depth == 101:
        model = ResNet(Bottleneck, [3, 4, 23, 3], network_type, num_classes, att_type, in_channel)
        if pretrained:
            import torchvision.models as models
            pt_model = models.resnet101(pretrained=True)
        else:
            pass

    if pretrained:
        model_dict = model.state_dict()
        pt_dict = pt_model.state_dict()
        for key in pt_dict.keys():
            if model_dict[key].shape != pt_dict[key].shape:
                if 'conv1' in key:
                    print(f"Size Mismatch in {key}, Reconfiguring...")
                    npt = torch.mean(pt_dict[key], dim=1, keepdim=True)
                    if model_dict[key].shape != npt.shape: 
                        print(f"Size Mismatch in {key}, skipping...")
                    else:
                        model_dict[key] = npt
                else:
                    print(f"Size Mismatch in {key}, skipping...")
            else:
                model_dict[key] = pt_dict[key]
        nn.init.xavier_uniform_(model_dict['fc.weight'], .1)
        nn.init.constant_(model_dict['fc.bias'], 0.)
        model.load_state_dict(model_dict, strict=False)
    else:
        pass
    return model
