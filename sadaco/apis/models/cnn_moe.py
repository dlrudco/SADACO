from typing import Any, Dict, List, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

__all__ = ["cnn_moe"]

CFG: Dict[str, List[Union[str, int, float]]] = {
    "conv1": [(1, 64), "AP", 0.10],
    "conv2": [(64, 128), "AP", 0.15],
    "conv3": [(128, 256), None, 0.20],
    "conv4": [(256, 256), "AP", 0.20],
    "conv5": [(256, 512), None, 0.25],
    "conv6": [(512, 512), None, None],
}


class DCNN(nn.Module):
    r"""DCNN model for VGG7-like architecture in CNN-MoE paper.

    Note: This implementation is based on details in Section V. Enhanced Deep Learning Framework, A. CNN-MoE Network architecture. Therefore the output of this DCNN is a 512-dim vector from the final global average pooling layer (w/o dropout and fc layer), which will be presented simultaneously to all experts.

    Args:
        features: A sequential module which may contains `nn.Conv2d`, `nn.BatchNorm2d`, `nn.AvgPool2d`, `nn.ReLU`, `nn.Dropout2d`, etc.
    """

    def __init__(
        self,
        features: nn.Sequential,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.features = features
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        return x


class CNN_MoE(nn.Module):
    r"""CNN-MoE model from `DCNN` and Mixture of Experts.

    Args:
        cnn: A `DCNN` model.
        num_classes: Number of classes.
        num_experts: Number of experts. (default: 10)
    """

    def __init__(
        self,
        cnn: DCNN,
        num_classes: int,
        num_experts: int = 10,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.cnn = cnn
        self.experts = nn.ModuleList(
            [
                nn.Sequential(nn.Linear(512, num_classes), nn.ReLU(inplace=True))
                for _ in range(num_experts)
            ]
        )
        self.softmax_gate = nn.Linear(512, num_experts)

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cnn(x)
        gate = F.softmax(self.softmax_gate(x), dim=1)
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)
        gate_expert_outputs = torch.einsum("b e, b e c -> b e c", gate, expert_outputs)
        gate_outputs_sum = torch.sum(gate_expert_outputs, dim=1)
        return gate_outputs_sum


def make_block(
    block_cfg: Dict[str, List[Union[str, int, float]]] = CFG
) -> nn.Sequential:
    r"""Create conv block for VGG7-like architecture.

    Args:
        block_cfg: Block configuration. (default: CFG)
    """
    block: List[nn.Module] = []
    for _, layer_cfg in block_cfg.items():
        for value in layer_cfg:
            if value == "AP":
                block += [nn.AvgPool2d(kernel_size=(2, 2))]
            elif type(value) == tuple:
                conv2d = nn.Conv2d(
                    in_channels=value[0],
                    out_channels=value[1],
                    kernel_size=(3, 3),
                    padding=1,
                )
                block += [
                    nn.BatchNorm2d(num_features=value[0]),
                    conv2d,
                    nn.ReLU(inplace=True),
                    nn.BatchNorm2d(num_features=value[1]),
                ]
            elif type(value) == float:
                block += [nn.Dropout2d(p=value)]
    return nn.Sequential(*block)


def _dcnn(cfg: Dict[str, List[Union[str, int, float]]] = CFG, **kwargs: Any) -> DCNN:
    r"""Create DCNN model for VGG7-like architecture.

    Args:
        cfg: Block configuration. (default: CFG)
    """
    features = make_block(cfg)
    return DCNN(features, **kwargs)


def cnn_moe(
    num_classes: int,
    num_experts: int = 10,
    cfg: Dict[str, List[Union[str, int, float]]] = CFG,
    **kwargs: Any,
) -> CNN_MoE:
    r"""Create CNN-MoE model from `DCNN` and Mixture of Experts.

    Args:
        num_classes: Number of classes.
        num_experts: Number of experts. (default: 10)
        cfg: Block configuration. (default: CFG)
    """
    return CNN_MoE(_dcnn(cfg), num_classes, num_experts, **kwargs)


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = cnn_moe(num_classes=4, num_experts=10).to(device)
    summary(model, (1, 64, 64))
