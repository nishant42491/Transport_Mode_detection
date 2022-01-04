import torch
from torch import nn
from torch.autograd import Variable


class ModalityLSTM(nn.Module):
    def __init__(self, trip_dimension, output_size, batch_size, hidden_dim, n_layers, gpu, drop_prob, lstm_drop_prob=0.5):
        super().__init__()
        self.trip_dimension = trip_dimension
        self.output_size = output_size
        self.n_layers = n_layers
        self.batch_size = batch_size
        self.hidden_dim = hidden_dim
        self.on_gpu = gpu
        self.lstm_drop_prob = lstm_drop_prob
        self.drop_prob = drop_prob

        self.lstm = nn.LSTM(
            input_size=self.trip_dimension,
            hidden_size=self.hidden_dim,
            num_layers=self.n_layers,
            batch_first=True,
            dropout = self.drop_prob,
            bidirectional=True
        )
        self.dropout = nn.Dropout(drop_prob)
        self.linear_fc = nn.Linear(self.hidden_dim * 2, self.output_size)


    def init_hidden(self):
        # the weights are of the form (nb_layers, batch_size, nb_lstm_units)
        hidden_a = torch.randn(self.n_layers*2, self.batch_size, self.hidden_dim)
        hidden_b = torch.randn(self.n_layers*2, self.batch_size, self.hidden_dim)

        if self.on_gpu:
            hidden_a = hidden_a.cuda()
            hidden_b = hidden_b.cuda()

        hidden_a = Variable(hidden_a)
        hidden_b = Variable(hidden_b)

        return (hidden_a, hidden_b)

    def forward(self, input_tensor, lengths):
        # shape of X: [batch_size, max_seq_len, feature_size]

        # get unpadded sequence lengths (padding: -1)
        self.hidden = self.init_hidden()

        # pack the padded sequences, length contains unpadded lengths (eg., [43,46,67,121])
        x_packed = torch.nn.utils.rnn.pack_padded_sequence(input_tensor, lengths, batch_first=True)

        # feed to lstm
        lstm_out, self.hidden = self.lstm(x_packed.float(), self.hidden)

        # unpack
        x_unpacked, seq_len = torch.nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)

        out = self.dropout(x_unpacked)

        outs = []  # save all predictions
        for point in out:
            outs.append(self.linear_fc(point))
        return torch.stack(outs, dim=0),max(lengths)