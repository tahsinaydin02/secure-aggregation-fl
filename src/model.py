"""CNN architecture used across all three phases of the project.

Design choice: the model must be *twice differentiable*
-------------------------------------------------------
DLG (Zhu et al., 2019) reconstructs an input by minimizing

    L_attack(x', y') = || grad_W L_train(x', y') - g_target ||^2

where g_target is the gradient the victim sent to the server. Because
L_attack already contains a gradient of L_train, the attacker's optimizer
needs to differentiate L_attack with respect to (x', y') — i.e. it needs
*second derivatives* of L_train with respect to the model weights.

ReLU is non-differentiable at 0, which makes second-order derivatives
undefined at exactly that point. In practice PyTorch returns 0 there, but
this corrupts the optimization landscape and DLG often fails to converge.

The DLG paper sidesteps this by using **Sigmoid** activations. We follow
the same recipe. This is documented in the report's Methodology section
as a deliberate compromise: we trade off some baseline accuracy (Sigmoid
trains slower than ReLU) for a model on which the attack actually works,
because the *point* of Phase 2 is to demonstrate that the attack works
under realistic assumptions.

Architecture
------------
A small CNN, similar in spirit to LeNet, with two conv-pool blocks and
one fully-connected head. Roughly 80k parameters — small enough that DLG
optimization converges in a few hundred iterations on a laptop CPU, but
non-trivial enough that the exercise is not vacuous.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MNIST_CNN(nn.Module):
    """Small CNN for MNIST classification.

    Input  : (B, 1, 28, 28)   normalized grayscale images
    Output : (B, 10)          unnormalized logits over the 10 digit classes

    Activations are Sigmoid throughout to ensure twice-differentiability
    (see module docstring for why). MaxPool is replaced with AvgPool for
    the same reason — max is a piecewise function whose second derivative
    is ill-defined at the argmax boundary.
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()

        # Block 1: 1 -> 12 channels, 5x5 kernel, padding=2 keeps spatial dim.
        # 28x28 -> conv -> 28x28 -> avg-pool 2x2 -> 14x14.
        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=12, kernel_size=5, padding=2,
        )

        # Block 2: 12 -> 12 channels, 5x5, padding=2.
        # 14x14 -> conv -> 14x14 -> avg-pool 2x2 -> 7x7.
        self.conv2 = nn.Conv2d(
            in_channels=12, out_channels=12, kernel_size=5, padding=2,
        )

        # Fully-connected head.
        # Flattened feature map size: 12 * 7 * 7 = 588.
        self.fc = nn.Linear(in_features=12 * 7 * 7, out_features=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Block 1
        x = self.conv1(x)
        x = torch.sigmoid(x)              # twice-differentiable activation
        x = F.avg_pool2d(x, kernel_size=2)  # smooth alternative to MaxPool

        # Block 2
        x = self.conv2(x)
        x = torch.sigmoid(x)
        x = F.avg_pool2d(x, kernel_size=2)

        # Flatten everything except the batch dim and run through the head.
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        # We return raw logits — the caller pairs this with
        # nn.CrossEntropyLoss, which applies log_softmax internally.
        return x


def num_parameters(model: nn.Module) -> int:
    """Count trainable parameters — handy for the report's model description."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Smoke test — verify shapes and parameter count.
    model = MNIST_CNN()
    print(model)
    print(f"Trainable parameters: {num_parameters(model):,}")
    dummy = torch.randn(4, 1, 28, 28)  # batch of 4
    out = model(dummy)
    print(f"Output shape: {tuple(out.shape)}")  # expect (4, 10)
