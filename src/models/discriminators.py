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

class ECGConvMinibatchDiscriminator(nn.Module):
  def __init__(self,seq_length,batch_size,minibatch_normal_init, n_features = 1, num_cv = 1, minibatch = 0, cv1_out= 10, cv1_k = 3, cv1_s = 4, p1_k = 3, p1_s = 3, cv2_out = 10, cv2_k = 3, cv2_s = 3 ,p2_k = 3, p2_s = 3):
      super(Discriminator,self).__init__()
      self.n_features = n_features
      self.seq_length = seq_length
      self.batch_size = batch_size
      self.num_cv = num_cv
      self.minibatch = minibatch
      self.cv1_dims = int((((((seq_length - cv1_k)/cv1_s) + 1)-p1_k)/p1_s)+1)
      self.cv2_dims = int((((((self.cv1_dims - cv2_k)/cv2_s) + 1)-p2_k)/p2_s)+1)
      self.cv1_out = cv1_out
      self.cv2_out = cv2_out

      #input should be size (batch_size,num_features,seq_length) for the convolution layer
      self.CV1 = nn.Sequential(
                  nn.Conv1d(in_channels = self.n_features, out_channels = int(cv1_out),kernel_size = int(cv1_k), stride = int(cv1_s))
                  ,nn.ReLU()
                  ,nn.MaxPool1d(kernel_size = int(p1_k), stride = int(p1_s))
                 )

      # 2 convolutional layers
      if self.num_cv > 1:
        self.CV2 = nn.Sequential(
                      nn.Conv1d(in_channels = int(cv1_out), out_channels = int(cv2_out) ,kernel_size =int(cv2_k), stride = int(cv2_s))
                      ,nn.ReLU()
                      ,nn.MaxPool1d(kernel_size = int(p2_k), stride = int(p2_s))
                  )

        #Adding a minibatch discriminator layer to add a cripple affect to the discriminator so that it needs to generate sequences that are different from each other.

        if   self.minibatch > 0:
          self.mb1 = MinibatchDiscrimination(self.cv2_dims*cv2_out,self.minibatch, minibatch_normal_init)
          self.out = nn.Sequential(nn.Linear(int(self.cv2_dims*cv2_out)+self.minibatch,1),nn.Sigmoid()) # to make sure the output is between 0 and 1
        else:
          self.out = nn.Sequential(nn.Linear(int(self.cv2_dims*cv2_out),1),nn.Sigmoid()) # to make sure the output is between 0 and 1

      # 1 convolutional layer
      else:

        #Adding a minibatch discriminator layer to add a cripple affect to the discriminator so that it needs to generate sequences that are different from each other.
        if self.minibatch > 0 :

          self.mb1 = MinibatchDiscrimination(int(self.cv1_dims*cv1_out),self.minibatch, minibatch_normal_init)
          self.out = nn.Sequential(nn.Linear(int(self.cv1_dims*cv1_out)+self.minibatch,1),nn.Dropout(0.2),nn.Sigmoid()) # to make sure the output is between 0 and 1
        else:
          self.out = nn.Sequential(nn.Linear(int(self.cv1_dims*cv1_out),1),nn.Sigmoid())

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
