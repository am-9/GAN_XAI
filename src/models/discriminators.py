"""
This file contains the discriminator architectures.

Date:
    August 15, 2020

Project:
    XAI-GAN

Contact:
    explainable.gan@gmail.com
"""

from abc import ABC
from torch import nn, Tensor
import numpy as np

class EcgDiscriminator(nn.Module):
    def __init__(self):
        super(EcgDiscriminator, self).__init__()
        ndf = 64
        self.main = nn.Sequential(
        # input is (nc) x 64 x 64
        nn.Conv1d(in_channels=1, out_channels=ndf, kernel_size=4, stride=2, padding=1, bias=False),
        nn.LeakyReLU(0.2, inplace=True),
        # state size. (ndf) x 32 x 32
        nn.Conv1d(ndf, ndf * 2, 4, 2, 1, bias=False),
        nn.BatchNorm1d(ndf * 2),
        nn.LeakyReLU(0.2, inplace=True),
        # state size. (ndf*2) x 16 x 16
        nn.Conv1d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
        nn.BatchNorm1d(ndf * 4),
        nn.LeakyReLU(0.2, inplace=True),
        # state size. (ndf*4) x 8 x 8
        nn.Conv1d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
        nn.BatchNorm1d(ndf * 8),
        nn.LeakyReLU(0.2, inplace=True),

        nn.Conv1d(ndf * 8, ndf * 16, 4, 2, 1, bias=False),
        nn.BatchNorm1d(ndf * 16),
        nn.LeakyReLU(0.2, inplace=True),

        # state size. (ndf*8) x 4 x 4
        nn.Conv1d(ndf * 16, 1, 5, 2, 0, bias=False),
        nn.Sigmoid()
        )

    def forward(self, x):
        x = x.view(-1, 1, 216)
        return self.main(x)

class DiscriminatorNet(nn.Module, ABC):
    """
    A simple three hidden-layer discriminative neural network
    """
    def __init__(self):
        super(DiscriminatorNet, self).__init__()
        self.n_features = (1, 32, 32)

        self.input_layer = nn.Sequential(
            nn.Linear(int(np.prod(self.n_features)), 1296),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )
        self.hidden1 = nn.Sequential(
            nn.Linear(1296, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )
        self.hidden2 = nn.Sequential(
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )
        self.out = nn.Sequential(
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x: Tensor) -> Tensor:
        """ overrides the __call__ method of the discriminator """
        x = x.view(x.size(0), -1)
        x = self.input_layer(x)
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.out(x)
        return x


class DiscriminatorNetCifar10(nn.Module, ABC):
    def __init__(self):
        super(DiscriminatorNetCifar10, self).__init__()
        self.n_features = (3, 32, 32)
        nc, ndf = 3, 64

        self.input_layer = nn.Sequential(
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.hidden1 = nn.Sequential(
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.hidden2 = nn.Sequential(
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.out = nn.Sequential(
            nn.Conv2d(ndf * 4, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        """ overrides the __call__ method of the discriminator """
        x = self.input_layer(x)
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.out(x)
        return x
