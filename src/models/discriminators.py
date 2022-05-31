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
from torch import nn, Tensor, jit, randn, exp, cat, stack
import numpy as np
import torch
import torch.nn.init as init

class IMVTensorLSTM(torch.jit.ScriptModule):

    __constants__ = ["n_units", "input_dim"]
    def __init__(self, input_dim=1, output_dim=1, n_units=100, init_std=0.02):
        super().__init__()
        self.U_j = nn.Parameter(torch.randn(input_dim, 1, n_units)*init_std)
        self.U_i = nn.Parameter(torch.randn(input_dim, 1, n_units)*init_std)
        self.U_f = nn.Parameter(torch.randn(input_dim, 1, n_units)*init_std)
        self.U_o = nn.Parameter(torch.randn(input_dim, 1, n_units)*init_std)
        self.W_j = nn.Parameter(torch.randn(input_dim, n_units, n_units)*init_std)
        self.W_i = nn.Parameter(torch.randn(input_dim, n_units, n_units)*init_std)
        self.W_f = nn.Parameter(torch.randn(input_dim, n_units, n_units)*init_std)
        self.W_o = nn.Parameter(torch.randn(input_dim, n_units, n_units)*init_std)
        self.b_j = nn.Parameter(torch.randn(input_dim, n_units)*init_std)
        self.b_i = nn.Parameter(torch.randn(input_dim, n_units)*init_std)
        self.b_f = nn.Parameter(torch.randn(input_dim, n_units)*init_std)
        self.b_o = nn.Parameter(torch.randn(input_dim, n_units)*init_std)
        self.F_alpha_n = nn.Parameter(torch.randn(input_dim, n_units, 1)*init_std)
        self.F_alpha_n_b = nn.Parameter(torch.randn(input_dim, 1)*init_std)
        self.F_beta = nn.Linear(2*n_units, 1)
        self.Phi = nn.Linear(2*n_units, output_dim)
        self.n_units = n_units
        self.input_dim = input_dim

    @torch.jit.script_method
    def forward(self, test):
        x = torch.unsqueeze(test, dim=-1)
        #print("legnth of shape ", len(x.shape))
        if len(x.shape) == 2:
            x = torch.unsqueeze(x, dim=-1)
        #print ("x shape ", x.shape)

        h_tilda_t = torch.zeros(x.shape[0], self.input_dim, self.n_units)
        c_tilda_t = torch.zeros(x.shape[0], self.input_dim, self.n_units)
        outputs = torch.jit.annotate(List[Tensor], [])
        for t in range(x.shape[1]):
            # eq 1
            j_tilda_t = torch.tanh(torch.einsum("bij,ijk->bik", h_tilda_t, self.W_j) + \
                                   torch.einsum("bij,jik->bjk", x[:,t,:].unsqueeze(1), self.U_j) + self.b_j)
            # eq 5
            i_tilda_t = torch.sigmoid(torch.einsum("bij,ijk->bik", h_tilda_t, self.W_i) + \
                                torch.einsum("bij,jik->bjk", x[:,t,:].unsqueeze(1), self.U_i) + self.b_i)
            f_tilda_t = torch.sigmoid(torch.einsum("bij,ijk->bik", h_tilda_t, self.W_f) + \
                                torch.einsum("bij,jik->bjk", x[:,t,:].unsqueeze(1), self.U_f) + self.b_f)
            o_tilda_t = torch.sigmoid(torch.einsum("bij,ijk->bik", h_tilda_t, self.W_o) + \
                                torch.einsum("bij,jik->bjk", x[:,t,:].unsqueeze(1), self.U_o) + self.b_o)
            # eq 6
            c_tilda_t = c_tilda_t*f_tilda_t + i_tilda_t*j_tilda_t
            # eq 7
            h_tilda_t = (o_tilda_t*torch.tanh(c_tilda_t))
            outputs += [h_tilda_t]
        outputs = torch.stack(outputs)
        outputs = outputs.permute(1, 0, 2, 3)
        # eq 8
        alphas = torch.tanh(torch.einsum("btij,ijk->btik", outputs, self.F_alpha_n) +self.F_alpha_n_b)
        alphas = torch.exp(alphas)
        alphas = alphas/torch.sum(alphas, dim=1, keepdim=True)
        g_n = torch.sum(alphas*outputs, dim=1)
        hg = torch.cat([g_n, h_tilda_t], dim=2)
        mu = self.Phi(hg)
        betas = torch.tanh(self.F_beta(hg))
        betas = torch.exp(betas)
        betas = betas/torch.sum(betas, dim=1, keepdim=True)
        mean = torch.sum(betas*mu, dim=1)

        return mean, alphas, betas

class MinibatchDiscrimination(nn.Module):
    def __init__(self, in_features, out_features, kernel_dims=16, mean=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.kernel_dims = kernel_dims
        self.mean = mean
        self.T = nn.Parameter(torch.Tensor(in_features, out_features, kernel_dims))
        init.normal(self.T, 0, 1)

    def forward(self, x):
        # x is NxA
        # T is AxBxC
        matrices = x.mm(self.T.view(self.in_features, -1))
        matrices = matrices.view(-1, self.out_features, self.kernel_dims)

        M = matrices.unsqueeze(0)  # 1xNxBxC
        M_T = M.permute(1, 0, 2, 3)  # Nx1xBxC
        norm = torch.abs(M - M_T).sum(3)  # NxNxB
        expnorm = torch.exp(-norm)
        o_b = (expnorm.sum(0) - 1)   # NxB, subtract self distance
        if self.mean:
            o_b /= x.size(0) - 1

        x = torch.cat([x, o_b], 1)
        return x

class EcgCNNDiscriminator(nn.Module):
    def __init__(self):
        super(EcgCNNDiscriminator, self).__init__()
        ndf = 64
        self.input = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv1d(in_channels=1, out_channels=ndf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.layer1 = nn.Sequential(
            # state size. (ndf) x 32 x 32
            nn.Conv1d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.InstanceNorm1d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.layer2 = nn.Sequential(
            # state size. (ndf*2) x 16 x 16
            nn.Conv1d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.InstanceNorm1d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.layer3 = nn.Sequential(
            # state size. (ndf*4) x 8 x 8
            nn.Conv1d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.InstanceNorm1d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.layer4 = nn.Sequential(
            nn.Conv1d(ndf * 8, ndf * 16, 4, 2, 1, bias=False),
            nn.InstanceNorm1d(ndf * 16),
            nn.LeakyReLU(0.2, inplace=True),
        )

        #self.mb1 = MinibatchDiscrimination(16*6, 0, kernel_dims=16)

        self.layer5 = nn.Sequential(
            # state size. (ndf*8) x 4 x 4
            nn.Conv1d(ndf * 16, 1, 5, 2, 0, bias=False),
        )

    def forward(self, x):
        #print ("initial shape ", x.shape)
        x = x.view(-1, 1, 216)
        # print ("at discriminator")
        # print ("initial shape ", x.shape)
        x = self.input(x)
        #print ("input shape ", x.shape)
        x = self.layer1(x)
        #print ("layer1 shape ", x.shape)
        x = self.layer2(x)
        #print ("layer2 shape ", x.shape)
        x = self.layer3(x)
        #print ("layer3 shape ", x.shape)
        x = self.layer4(x)
        #print ("layer4 shape ", x.shape)
        x = self.layer5(x)
        #x = torch.sigmoid(x)
        # print ("layer5 shape ", x.shape)
        # exit(0)
        return x

class ECGLSTMDiscriminator(nn.Module):
    def __init__(self, input_dim=216, hidden_dim=512, output_dim=1, num_layers=2):
        super(ECGLSTMDiscriminator, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.layer1 = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers, bidirectional=False, dropout=0.5)
        #self.mb1 = MinibatchDiscrimination(self.hidden_dim, self.hidden_dim)
        self.out = nn.Sequential(
            nn.Linear(self.hidden_dim, self.output_dim),
        )

    def forward(self, x):
        # s, b, dim
        #print ("at discrißminator")
        x = x.unsqueeze(0)
        #print ("shape of x ", x.shape)
        if len(x.shape) == 2:
            x = x.unsqueeze(0)
        #print ("shape of x ", x.shape)

        x, hn = self.layer1(x)
        s, b, h = x.size()
        x = x.view(s*b, h)
        #print ("shape after layer 1 ", x.shape)
        #x = self.mb1(x)
        #print ("shape after mb 1 ", x.shape)
        x = self.out(x)
        #x = torch.sigmoid(x)
        #exit(0)
        # s, b, outputsize
        #x = x.view(s, b, -1)
        return x

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
