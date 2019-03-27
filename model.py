import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class EncoderRNN(nn.Module):
    def __init__(self, weight_matrix, embedding_size, hidden_size, bidirectional):
        super(EncoderRNN, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(weight_matrix, freeze=True)
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.gru = nn.GRU(embedding_size, hidden_size, num_layers=2, bidirectional=self.bidirectional, dropout=0.1)

    def forward(self, input_t, hidden=None):
        embedded = self.embedding(input_t)
        outputs, hidden = self.gru(embedded, hidden)
        if self.bidirectional:
            outputs = outputs[:, :, :self.hidden_size] + outputs[:, : ,self.hidden_size:]
        return outputs, hidden

    def initHidden(self, batch_size):
        if self.bidirectional:
            return torch.zeros(4, batch_size, self.hidden_size, device=device)
        return torch.zeros(2, batch_size, self.hidden_size, device=device)

class Attn(nn.Module):
    def __init__(self, hidden_size):
        super(Attn, self).__init__()
        self.hidden_size = hidden_size
        self.attn = nn.Linear(self.hidden_size, hidden_size)

    def forward(self, hidden, encoder_outputs):
        max_len = encoder_outputs.size(0)
        this_batch_size = encoder_outputs.size(1)
        # Create variable to store attention energies
        attn_energies = Variable(torch.zeros(this_batch_size, max_len)).to(device) # B x 1 x S
        # Calculate energies for each batch of encoder outputs
        for b in range(this_batch_size):
            # Calculate energy for each encoder output
            for i in range(max_len):
                attn_energies[b, i] = self.score(hidden[:, b].squeeze(0), encoder_outputs[i, b].unsqueeze(0))
        # Normalize energies to weights in range 0 to 1, resize to 1 x B x S
        return F.softmax(attn_energies).unsqueeze(1)

    def score(self, hidden, encoder_output):
        energy = self.attn(encoder_output)
        energy = hidden.dot(energy.squeeze(0))
        return energy

class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size,  n_layers=2, dropout_p=0.1):
        super(AttnDecoderRNN, self).__init__()

        # Define parameters
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p

        # Define layers
        self.dropout = nn.Dropout(dropout_p)
        self.attn = Attn(hidden_size).to(device)
        self.pre_linear = nn.Linear(hidden_size + 10, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=dropout_p)
        self.post_linear = nn.Linear(hidden_size, output_size)

    def forward(self, motion_input, last_hidden, encoder_outputs):
        """
        """
        attn_weights = self.attn(last_hidden[-1].unsqueeze(0), encoder_outputs)
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1)) # B x 1 x N

        rnn_input = torch.cat((motion_input.unsqueeze(1), context), 2).to(device)
        rnn_input = self.pre_linear(rnn_input)
        output, hidden = self.gru(rnn_input.transpose(0, 1), last_hidden)
        output = self.post_linear(output)
        output = output.squeeze(0) # B x N
        return output, hidden, attn_weights
