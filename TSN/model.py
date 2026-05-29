"""Temporal Segment Network models for miniUCF."""

from __future__ import annotations

import torch
from torch import nn
from torchvision import models


NUM_CLASSES = 25
NUM_SEGMENTS = 4
FLOW_STACK_SIZE = 7
FLOW_CHANNELS = 2 * FLOW_STACK_SIZE


def pair(value) -> tuple[int, int]:
    if isinstance(value, tuple):
        return int(value[0]), int(value[1])
    return int(value), int(value)


def resnet18(pretrained: bool) -> nn.Module:
    """Create ResNet-18 while supporting old and new torchvision APIs."""

    try:
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        return models.resnet18(weights=weights)
    except AttributeError:
        return models.resnet18(pretrained=pretrained)


class TSN(nn.Module):
    """A simple Temporal Segment Network with a ResNet-18 backbone.

    The input shape is ``[batch, segments, channels, height, width]``.
    ResNet-18 is applied to each segment independently, and segment logits are
    averaged to obtain one video-level prediction.
    """

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        num_segments: int = NUM_SEGMENTS,
        input_channels: int = 3,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        self.num_segments = num_segments
        self.input_channels = input_channels

        self.backbone = resnet18(pretrained=pretrained)
        if input_channels != 3:
            self._replace_first_conv(input_channels=input_channels, pretrained=pretrained)

        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, segments, channels, height, width = x.shape
        if segments != self.num_segments:
            raise ValueError(f"Expected {self.num_segments} segments, got {segments}")
        if channels != self.input_channels:
            raise ValueError(f"Expected {self.input_channels} input channels, got {channels}")

        x = x.reshape(batch_size * segments, channels, height, width)
        logits = self.backbone(x)
        logits = logits.reshape(batch_size, segments, -1)
        return logits.mean(dim=1)

    def _replace_first_conv(self, input_channels: int, pretrained: bool) -> None:
        old_conv = self.backbone.conv1
        new_conv = nn.Conv2d(
            input_channels,
            old_conv.out_channels,
            kernel_size=pair(old_conv.kernel_size),
            stride=pair(old_conv.stride),
            padding=pair(old_conv.padding),
            bias=False,
        )

        if pretrained:
            with torch.no_grad():
                rgb_average = old_conv.weight.mean(dim=1, keepdim=True)
                new_conv.weight.copy_(rgb_average.repeat(1, input_channels, 1, 1))
                new_conv.weight.mul_(3.0 / float(input_channels))
        else:
            nn.init.kaiming_normal_(new_conv.weight, mode="fan_out", nonlinearity="relu")

        self.backbone.conv1 = new_conv


def rgb_tsn(pretrained: bool = True) -> TSN:
    return TSN(input_channels=3, pretrained=pretrained)


def flow_tsn(pretrained: bool = True) -> TSN:
    return TSN(input_channels=FLOW_CHANNELS, pretrained=pretrained)


__all__ = [
    "FLOW_CHANNELS",
    "FLOW_STACK_SIZE",
    "NUM_CLASSES",
    "NUM_SEGMENTS",
    "TSN",
    "flow_tsn",
    "rgb_tsn",
]
