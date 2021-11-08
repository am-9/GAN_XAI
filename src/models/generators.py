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

class EcgGenerator(nn.Module, ABC):
    def __init__(self):
        super(EcgGenerator, self).__init__()
        ngf = 64
        self.main = nn.Sequential(
            # shape in = [N, 50, 1]
            nn.ConvTranspose1d(100, ngf * 32, 4, 1, 0, bias=False),
            nn.BatchNorm1d(ngf * 32),
            nn.ReLU(True),
            # shape in = [N, 64*4, 4]
            nn.ConvTranspose1d(ngf * 32, ngf * 16, 4, 1, 0, bias=False),
            nn.BatchNorm1d(ngf * 16),
            nn.ReLU(True),
            # shape in = [N, 64*2, 7]
            nn.ConvTranspose1d(ngf * 16, ngf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm1d(ngf * 8),
            nn.ReLU(True),
            nn.ConvTranspose1d(ngf * 8, ngf * 4, 3, 2, 1, bias=False),
            nn.BatchNorm1d(ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose1d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm1d(ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose1d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm1d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose1d(ngf, 1, 4, 2, 1, bias=False),
        )

    def forward(self, x):
        x = x.view(-1, 100, 1)
        x = self.main(x)
        x = x.view(-1, 216)
        return x

class ECGLSTMGenerator(nn.Module):
    def __init__(self,seq_length,batch_size,n_features = 1, hidden_dim = 50,
               num_layers = 2, tanh_output = False):
        super(Generator,self).__init__()
        self.n_features = n_features
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.seq_length = seq_length
        self.batch_size = batch_size
        self.tanh_output = tanh_output

        if IMV_LSTM == True:
            self.F_alpha_n = nn.Parameter(torch.randn(input_dim, n_units, 1)*init_std)
            self.F_alpha_n_b = nn.Parameter(torch.randn(input_dim, 1)*init_std)
            self.F_beta = nn.Linear(2*n_units, 1)


        self.layer1 = nn.LSTM(input_size = self.n_features, hidden_size = self.hidden_dim,
                                  num_layers = self.num_layers,batch_first = True#,dropout = 0.2,
                                 )
        if self.tanh_output == True:
            self.out = nn.Sequential(nn.Linear(self.hidden_dim,1),nn.Tanh()) # to make sure the output is between 0 and 1 - removed ,nn.Sigmoid()
        else:
            self.out = nn.Linear(self.hidden_dim,1)

    def init_hidden(self):
        weight = next(self.parameters()).data
        hidden = (weight.new(self.num_layers, self.batch_size, self.hidden_dim).zero_().to(device), weight.new(self.num_layers, self.batch_size, self.hidden_dim).zero_().to(device))
        return hidden

    def forward(self,x,hidden):

        x,hidden = self.layer1(x.view(self.batch_size,self.seq_length,1),hidden)

        x = self.out(x)

        return x #,hidden

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
