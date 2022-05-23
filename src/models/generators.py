"""
This file contains the generator architectures.

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
import torch
#from ecg_dataset_pytorch import scale_signal


class EcgCNNGenerator(nn.Module, ABC):
    def __init__(self):
        super(EcgCNNGenerator, self).__init__()
        ngf = 64

        self.layer1 = nn.Sequential(
            # shape in = [N, 50, 1]
            nn.ConvTranspose1d(100, ngf * 32, 4, 1, 0, bias=False),
            nn.InstanceNorm1d(ngf * 32),
            nn.ReLU(True)
        )

        self.layer2 = nn.Sequential(

            nn.ConvTranspose1d(ngf * 32, ngf * 16, 4, 1, 0, bias=False),
            nn.InstanceNorm1d(ngf * 16),
            nn.ReLU(True)
        )

        self.layer3 = nn.Sequential(
            # shape in = [N, 64*2, 7]
            nn.ConvTranspose1d(ngf * 16, ngf * 8, 4, 2, 1, bias=False),
            nn.InstanceNorm1d(ngf * 8),
            nn.ReLU(True),
        )

        self.layer4 = nn.Sequential(
            nn.ConvTranspose1d(ngf * 8, ngf * 4, 3, 2, 1, bias=False),
            nn.InstanceNorm1d(ngf * 4),
            nn.ReLU(True)
        )

        self.layer5 = nn.Sequential(
            nn.ConvTranspose1d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.InstanceNorm1d(ngf * 2),
            nn.ReLU(True)
        )

        self.layer6 = nn.Sequential(
            nn.ConvTranspose1d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.InstanceNorm1d(ngf),
            nn.ReLU(True),
        )

        self.out = nn.Sequential(
            nn.ConvTranspose1d(ngf, 1, 4, 2, 1, bias=False)
        )

    def forward(self, x):
        x = x.view(-1, 100, 1)
        #print ("at generator")
        #print("x shape ", x.shape)
        x = self.layer1(x)
        #print("x shape layer 1", x.shape)
        x = self.layer2(x)
        #print("x shape layer 2", x.shape)
        x = self.layer3(x)
        #print("x shape layer 3", x.shape)
        x = self.layer4(x)
        #print("x shape layer 4", x.shape)
        x = self.layer5(x)
        #print("x shape layer 5", x.shape)
        x = self.layer6(x)
        #print("x shape layer 6", x.shape)
        x = self.out(x)
        #x = (torch.sigmoid(x)*2)-0.5
        #print("x shape ", x.shape)
        x = x.view(-1, 216)
        #print(x)
        #exit(0)
        return x

class ECGLSTMGenerator(nn.Module):
    def __init__(self, input_dim=100, hidden_dim=50, output_dim=216, num_layers=2):
        super(ECGLSTMGenerator, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.layer1 = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers, dropout=0.2)
        self.out = nn.Linear(self.hidden_dim, self.output_dim)

    def forward(self, x):
        # s, b, dim
        x = x.unsqueeze(0)
        #print ("shape of x ", x.shape)
        x, hn = self.layer1(x)
        #print ("shape of x after layer 1 ", x.shape)
        s, b, h = x.size()
        x = x.view(s*b, h)
        x = self.out(x)
        #x = (torch.sigmoid(x)*2)-0.5
        #print ("shape of x after layer 2 ", x.shape)
        # s, b, outputsize
        #x = x.view(s, b, -1)
        return x

class GeneratorNet(nn.Module, ABC):
    """
    A three hidden-layer generative neural network
    """

    def __init__(self):
        super(GeneratorNet, self).__init__()
        self.n_features = 100
        self.n_out = (1, 32, 32)

        self.input_layer = nn.Sequential(
            nn.Linear(self.n_features, 256),
            nn.LeakyReLU(0.2)
        )
        self.hidden1 = nn.Sequential(
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2)
        )
        self.hidden2 = nn.Sequential(
            nn.Linear(512, 1296),
            nn.LeakyReLU(0.2)
        )
        self.out = nn.Sequential(
            nn.Linear(1296, int(np.prod(self.n_out))),
            nn.Tanh()
        )

    def forward(self, x: Tensor) -> Tensor:
        """ overrides the __call__ method of the generator """
        x = self.input_layer(x)
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.out(x)
        x = x.view(x.size(0), *self.n_out)
        return x


class GeneratorNetCifar10(nn.Module, ABC):
    def __init__(self):
        super(GeneratorNetCifar10, self).__init__()
        self.n_features = 100
        self.n_out = (3, 32, 32)
        nc, nz, ngf = 3, 100, 64

        self.input_layer = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
        )

        self.hidden1 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
        )

        self.hidden2 = nn.Sequential(
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
        )

        self.out = nn.Sequential(
            nn.ConvTranspose2d(ngf * 2, nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        """ overrides the __call__ method of the generator """
        x = self.input_layer(x)
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.out(x)
        return x
