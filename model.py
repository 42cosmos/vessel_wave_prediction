import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable


class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, seq_length, num_layers, num_classes, dropout=0.0):
        super(GRUModel, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.seq_length = seq_length

        self.relu = nn.ReLU()
        self.gru = nn.LSTM(input_size=input_size,
                           hidden_size=hidden_size,
                           num_layers=num_layers,
                           dropout=dropout,
                           batch_first=True)
        self.fc_1 = nn.Linear(hidden_size, 128)
        self.fc = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size).cuda())
        c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size).cuda())

        out, (h_n, c_n) = self.gru(x, (h_0, c_0))

        h_n = h_n.view(-1, self.hidden_size)
        out = self.relu(h_n)
        out = self.fc_1(out)
        out = self.relu(out)
        out = self.fc(out)
        return out
