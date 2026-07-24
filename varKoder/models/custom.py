"""Custom (non-timm) architectures for varKoder, plus a factory that
materializes their LazyLinear layers so weights can be loaded from safetensors."""

import torch
from torch.nn import (
    Module, Sequential, Linear, Flatten, LazyLinear, ReLU, Dropout,
    Conv1d, MaxPool1d,
)


class Arias2022Head(Module):
    def __init__(self, n_classes):
        super().__init__()
        self.head = Sequential(Linear(64, n_classes))

    def forward(self, x):
        return self.head(x)


class Arias2022Body(Module):
    def __init__(self):
        super().__init__()
        self.body = Sequential(
            Flatten(), LazyLinear(512), ReLU(), Dropout(0.5),
            Linear(512, 64), ReLU(), Dropout(0.5),
        )

    def forward(self, x):
        x = x[:, 0, :, :]
        return self.body(x)


class Fiannaca2018Head(Module):
    def __init__(self, n_classes):
        super().__init__()
        self.head = Sequential(Linear(500, n_classes))

    def forward(self, x):
        return self.head(x)


class Fiannaca2018Body(Module):
    def __init__(self):
        super().__init__()
        self.flatten = Flatten()
        self.body = Sequential(
            Conv1d(1, 5, kernel_size=5), ReLU(), MaxPool1d(kernel_size=2),
            Conv1d(5, 10, kernel_size=5), ReLU(), MaxPool1d(kernel_size=2),
            Flatten(), LazyLinear(500), ReLU(),
        )

    def forward(self, x):
        x = x[:, 0, :, :]
        x = self.flatten(x)
        x = x.unsqueeze(1)
        return self.body(x)


class Fiannaca2018Model(Module):
    def __init__(self, n_classes):
        super().__init__()
        self.model = Sequential(Fiannaca2018Body(), Fiannaca2018Head(n_classes))

    def forward(self, x):
        return self.model(x)


class Arias2022Model(Module):
    def __init__(self, n_classes):
        super().__init__()
        self.model = Sequential(Arias2022Body(), Arias2022Head(n_classes))

    def forward(self, x):
        return self.model(x)


def instantiate_custom_model(architecture, num_classes, input_size):
    """Build a custom model and materialize its LazyLinear layers with a dummy
    forward at ``input_size`` (C, H, W), so its state_dict has concrete shapes."""
    if architecture == "arias2022":
        model = Arias2022Model(num_classes)
    elif architecture == "fiannaca2018":
        model = Fiannaca2018Model(num_classes)
    else:
        raise Exception("Custom models must be one of: fiannaca2018 arias2022")
    c, h, w = input_size
    with torch.no_grad():
        model(torch.randn(1, c, h, w))
    return model
