import torch.nn as nn
import logging

class ECGLSTM(nn.Module):
    def __init__(self, length_of_each_word, number_of_hidden_neurons,  num_of_classes, num_of_layers):
        super(ECGLSTM, self).__init__()

        self.len_of_word = length_of_each_word
        self.num_hidden_neurons = number_of_hidden_neurons
        self.output_size = num_of_classes
        self.num_of_layesr = num_of_layers
        self.lstm = nn.LSTM(length_of_each_word, number_of_hidden_neurons, num_of_layers, batch_first=True)
        self.output_layer = nn.Linear(number_of_hidden_neurons, num_of_classes)

    def forward(self, sentence):

        out, hidden = self.lstm(sentence)  # If (h_0, c_0) is not provided, both h_0 and c_0 default to zero.
        # last_output = out[-1]
        print ("OUT HIDDEN", out, hidden)
        last_output = out[:, -1, :]
        logging.debug("Shape of last output from LSTM: {}".format(last_output.shape))

        y_pred = self.output_layer(last_output)
        return y_pred
