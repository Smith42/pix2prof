""" 
A GRUNet in PyTorch

GRU reference:
    Kyunghyun Cho, Bart van Merrienboer, Caglar Gulcehre, Dzmitry Bahdanau,
    Fethi Bougares, Holger Schwenk, Yoshua Bengio.  
    Learning Phase Representations using RNN Encoder-Decoder for Statistical
    Machine Translation.
    arXiv:1406.1078
"""
import torch.nn as nn

class GRUNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, drop_prob=0.2):
        super(GRUNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.gru = nn.GRU(input_dim, hidden_dim, n_layers, bidirectional=False, batch_first=True, dropout=drop_prob)
        self.fc = nn.Linear(hidden_dim, output_dim) # *2 iff bidirectional == True
        self.relu = nn.ReLU()

    def forward(self, x, h):
        p, h = self.gru(x, h)
        # We take p's [:, 0]th value to get the first value in a sequence
        p = self.relu(self.fc(self.relu(p[:, 0])))
        return p, h
