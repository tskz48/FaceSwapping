import torch.nn as nn

import torch

import torch
import torch.nn as nn


class SpecificNorm(nn.Module):
    def __init__(self, epsilon=1e-8):
        super(SpecificNorm, self).__init__()

        self.register_buffer(
            "mean",
            torch.tensor(
                [0.485, 0.456, 0.406],
                dtype=torch.float32
            ).view(1, 3, 1, 1)
        )

        self.register_buffer(
            "std",
            torch.tensor(
                [0.229, 0.224, 0.225],
                dtype=torch.float32
            ).view(1, 3, 1, 1)
        )

    def forward(self, x):
        return (x - self.mean) / self.std