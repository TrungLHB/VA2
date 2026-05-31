"""3D ResNet-18 for RGB action recognition on miniUCF.

The architecture mirrors the 2D ResNet-18 layout, but every convolution is a
3D convolution.  When ``pretrained=True``, ImageNet ResNet-18 weights are
inflated by copying each 2D filter along the temporal dimension and dividing by
the temporal kernel size, following the I3D initialization idea from Carreira
and Zisserman (2017).
"""

from __future__ import annotations

from typing import Type

import torch
from torch import nn
from torchvision import models


NUM_CLASSES = 25


def triple(value: int | tuple[int, int, int]) -> tuple[int, int, int]:
    if isinstance(value, tuple):
        return int(value[0]), int(value[1]), int(value[2])
    return int(value), int(value), int(value)


def conv3x3x3(in_planes: int, out_planes: int, stride: int | tuple[int, int, int] = 1) -> nn.Conv3d:
    """3x3x3 convolution with padding."""

    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=triple(stride),
        padding=1,
        bias=False,
    )


def conv1x1x1(in_planes: int, out_planes: int, stride: int | tuple[int, int, int] = 1) -> nn.Conv3d:
    """1x1x1 convolution used for residual projection shortcuts."""

    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=1,
        stride=triple(stride),
        bias=False,
    )


class BasicBlock3D(nn.Module):
    """The 3D version of the basic residual block used by ResNet-18."""

    expansion = 1

    def __init__(
        self,
        in_planes: int,
        planes: int,
        stride: int | tuple[int, int, int] = 1,
        downsample: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.conv1 = conv3x3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = self.relu(out)
        return out


class ResNet3D(nn.Module):
    """3D equivalent of ResNet-18.

    Input tensors are expected in ``[batch, channels, frames, height, width]``
    format.  The default classifier predicts the 25 miniUCF classes.
    """

    def __init__(
        self,
        block: Type[BasicBlock3D],
        layers: list[int],
        num_classes: int = NUM_CLASSES,
        input_channels: int = 3,
        pretrained: bool = False,
    ) -> None:
        super().__init__()
        self.input_channels = input_channels
        self.in_planes = 64

        self.conv1 = nn.Conv3d(
            input_channels,
            self.in_planes,
            kernel_size=(3, 7, 7),
            stride=(1, 2, 2),
            padding=(1, 3, 3),
            bias=False,
        )
        self.bn1 = nn.BatchNorm3d(self.in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        if pretrained:
            self.inflate_resnet18_weights()

    def _make_layer(
        self,
        block: Type[BasicBlock3D],
        planes: int,
        blocks: int,
        stride: int | tuple[int, int, int] = 1,
    ) -> nn.Sequential:
        downsample = None
        if triple(stride) != (1, 1, 1) or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1x1(self.in_planes, planes * block.expansion, stride),
                nn.BatchNorm3d(planes * block.expansion),
            )

        layers = [block(self.in_planes, planes, stride, downsample)]
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, planes))

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 5:
            raise ValueError("Expected input shape [batch, channels, frames, height, width]")
        if x.size(1) != self.input_channels:
            raise ValueError(f"Expected {self.input_channels} input channels, got {x.size(1)}")

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)

    def inflate_resnet18_weights(self) -> None:
        """Initialize this 3D network from a 2D ImageNet ResNet-18."""

        resnet2d = _resnet18_imagenet()
        resnet2d_state = resnet2d.state_dict()

        own_state = self.state_dict()
        inflated_state = {}
        for name, parameter in own_state.items():
            if name.startswith("fc."):
                inflated_state[name] = parameter
                continue

            source = resnet2d_state.get(name)
            if source is None:
                inflated_state[name] = parameter
            elif parameter.ndim == 5 and source.ndim == 4:
                inflated_state[name] = _inflate_conv_weight(source, parameter.shape)
            elif parameter.shape == source.shape:
                inflated_state[name] = source
            else:
                inflated_state[name] = parameter

        self.load_state_dict(inflated_state)


def _inflate_conv_weight(weight_2d: torch.Tensor, target_shape: torch.Size) -> torch.Tensor:
    """Inflate a 2D conv kernel ``[out, in, h, w]`` to 3D ``[out, in, t, h, w]``."""

    temporal_size = target_shape[2]
    inflated = weight_2d.unsqueeze(2).repeat(1, 1, temporal_size, 1, 1)
    return inflated / float(temporal_size)


def _resnet18_imagenet() -> nn.Module:
    """Create an ImageNet ResNet-18 while supporting old torchvision APIs."""

    try:
        weights = models.ResNet18_Weights.IMAGENET1K_V1
        return models.resnet18(weights=weights)
    except AttributeError:
        return models.resnet18(pretrained=True)


def resnet3d18(num_classes: int = NUM_CLASSES, pretrained: bool = False) -> ResNet3D:
    """Build the RGB 3D ResNet-18 used for Task 2."""

    return ResNet3D(
        block=BasicBlock3D,
        layers=[2, 2, 2, 2],
        num_classes=num_classes,
        input_channels=3,
        pretrained=pretrained,
    )


def rgb_resnet3d18(pretrained: bool = False) -> ResNet3D:
    return resnet3d18(num_classes=NUM_CLASSES, pretrained=pretrained)


__all__ = [
    "BasicBlock3D",
    "NUM_CLASSES",
    "ResNet3D",
    "resnet3d18",
    "rgb_resnet3d18",
]
