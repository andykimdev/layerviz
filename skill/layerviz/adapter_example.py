"""
Give layerviz a model factory and a sample loader for a small torchvision style CNN.

Purpose:

- Build the model with build_model, optionally loading a checkpoint into it.
- Return one input batch with load_sample, shaped [1, 3, H, W] like a dataset image.

Special Notes:

- Copy this file to <project>/layerviz_adapter.py. In a real project, import the model class
  and the evaluation transform from the project instead of defining SmallResNet here.
- Run from the project root: python -m layerviz capture --model layerviz_adapter:build_model
  --sample layerviz_adapter:load_sample --out layerviz_runs/capture
"""

from pathlib import Path
from typing import Optional

import torch
from torch import nn

from layerviz.loading import load_state

IMAGE_SIZE = 64  # Small enough to capture on a CPU in seconds, large enough to show structure.


class BasicBlock(nn.Module):
    """
    Apply two 3x3 convolutions with a residual connection, as in torchvision ResNet.

    """

    def __init__(self, cin: int, cout: int, stride: int = 1) -> None:
        """Create the convolutions, the norms, and a 1x1 shortcut when the shape changes."""
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.relu = nn.ReLU()
        self.downsample = (
            nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))
            if stride != 1 or cin != cout else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return relu(block(x) + shortcut(x))."""
        out = self.relu(self.bn1(self.conv1(x)))
        return self.relu(self.bn2(self.conv2(out)) + self.downsample(x))


class SmallResNet(nn.Module):
    """
    Classify images with a stem, two residual stages, global pooling, and a linear head.

    """

    def __init__(self, num_classes: int = 10) -> None:
        """Create the layers with torchvision style names: stem, layer1, layer2, fc."""
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, 16, 3, 1, 1, bias=False), nn.BatchNorm2d(16),
                                  nn.ReLU())
        self.layer1 = BasicBlock(16, 16)
        self.layer2 = BasicBlock(16, 32, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(32, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return class logits of shape [B, num_classes] for images of shape [B, 3, H, W]."""
        x = self.layer2(self.layer1(self.stem(x)))
        return self.fc(torch.flatten(self.avgpool(x), 1))


def build_model(checkpoint: Optional[Path | str] = None) -> nn.Module:
    """
    Build the model and load a checkpoint into it when one is given.

    Args:
        checkpoint: A state dict file, a dict holding one under state_dict or model, or a
            saved nn.Module. None keeps the initial weights.

    Returns:
        The model in eval mode.

    Notes:
        python -m layerviz calls this with no arguments and loads --checkpoint itself.
        The argument serves Python callers such as layerviz.compare.
    """
    model = SmallResNet()
    if checkpoint is not None:
        state = load_state(checkpoint)
        model.load_state_dict(state.state_dict() if isinstance(state, nn.Module) else state)
    return model.eval()


def load_sample() -> torch.Tensor:
    """
    Return one image batch of shape [1, 3, IMAGE_SIZE, IMAGE_SIZE].

    Returns:
        A deterministic image of a bright disk on a color gradient, so layers have edges and
        regions to encode. In a real project, return the first evaluation image after the
        project's own transform, with a batch dimension added.
    """
    axis = torch.linspace(-1, 1, IMAGE_SIZE)
    y, x = torch.meshgrid(axis, axis, indexing="ij")
    disk = ((x - 0.2) ** 2 + (y + 0.1) ** 2 < 0.25).float()
    image = torch.stack([(x + 1) / 2, (y + 1) / 2, disk])
    return image.unsqueeze(0)
