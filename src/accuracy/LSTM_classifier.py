import torch.nn as nn
import logging

class ECGLSTM(nn.Module):
    def __init__(self, input_dim=216, hidden_dim=100, output_dim=1, num_layers=2):
        super(ECGLSTM, self).__init__()
        self.input_dim = input_dim
        print ('input dim', self.input_dim)
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.layer1 = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers, bidirectional=True, dropout=0.5)
        self.out = nn.Sequential(
            nn.Linear(self.hidden_dim*2, self.output_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        # s, b, dim
        print ("at classifer")
        x = x.unsqueeze(0)
        print ("shape of x ", x.shape)
        x, hn = self.layer1(x)
        s, b, h = x.size()
        x = x.view(s*b, h)
        x = self.out(x)
        # s, b, outputsize
        #x = x.view(s, b, -1)
        return x
