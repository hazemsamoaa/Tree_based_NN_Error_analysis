import torch
import torch.nn as nn
from torch.nn import functional as F


class Code2vecNet(nn.Module):
    def __init__(self, feature_size=384, label_size=1):
        super().__init__()

        self.feature_size = feature_size
        self.label_size = label_size

        # Latent layers
        self.h1 = nn.Linear(feature_size, feature_size)
        self.h2 = nn.Linear(feature_size, feature_size)

        # Hidden layer
        self.output = nn.Linear(feature_size, label_size)

    def pooling_layer(self, x):
        return torch.mean(x, dim=1)

    def forward(self, x):
        x = self.h1(x)
        x = self.pooling_layer(x)
        x = self.h2(x)
        output = self.output(x)
        return output
