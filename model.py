import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class EncoderRNN(nn.Module):
    def __init__(self, weight_matrix, input_size, hidden_size, bidirectional, n_layers=2, dropout=0.1):
        super(EncoderRNN, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.n_layers = n_layers
        self.dropout = dropout
        # lookup table from pre-trained embedding matrix (globe)        
        self.embedding = nn.Embedding.from_pretrained(weight_matrix, freeze=True)
        self.gru = nn.GRU(hidden_size+100, hidden_size, bidirectional=self.bidirectional, num_layers=n_layers, dropout=self.dropout)
        
    def forward(self, input_seqs, input_lengths, hidden=None):
        embedded = self.embedding(input_seqs)
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)
        output, hidden = self.gru(packed, hidden)
        output, _ = torch.nn.utils.rnn.pad_packed_sequence(output) # unpacked, backed to padded
        # output, hidden = self.gru(embedded, hidden)
        if self.bidirectional:
            output = output[:,:,:self.hidden_size] + output[:,:,self.hidden_size:]

        return output, hidden         


class Attn(nn.Module):
    def __init__(self, method, hidden_size):
        super(Attn, self).__init__()
        
        self.method = method
        self.hidden_size = hidden_size
        
        if self.method == 'general':
            self.attn = nn.Linear(self.hidden_size, hidden_size)
            
    def forward(self, hidden, encoder_outputs):
        seq_len = encoder_outputs.size(0)
        batch_size = encoder_outputs.size(1)
        
        # variable to store attention energies
        attn_energies = torch.zeros(batch_size, seq_len).to(device)
        
        # for each batch of encoder outputs
        for b in range(batch_size):
            # caculate energy for each encoder output
            for i in range(seq_len):
                attn_energies[b, i] = self.score(hidden[:,b].squeeze(0), encoder_outputs[i,b].unsqueeze(0))
                
        return F.softmax(attn_energies, dim=1).unsqueeze(1)
       
    def score(self, hidden, encoder_output):
        if self.method == 'general':
            energy = self.attn(encoder_output)
            energy = energy.squeeze(0)
            energy = hidden.dot(energy)
            return energy


class AttnDecoderRNN(nn.Module):
    def __init__(self, attn_model, hidden_size, output_size, n_layers=2, dropout=0.1):
        super(AttnDecoderRNN, self).__init__()
        
        self.attn_model = attn_model
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout = dropout
        
        self.attn = Attn(attn_model, hidden_size)
        self.pre_linear = nn.Linear(hidden_size + 10, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=dropout)
        self.post_linear = nn.Linear(hidden_size, output_size)
        
    def forward(self, motion_input, last_hidden, encoder_outputs):
        batch_size = motion_input.size(0)
        
        # calculate attention
        attn_weights = self.attn(last_hidden[-1].unsqueeze(0), encoder_outputs)
        context = attn_weights.bmm(encoder_outputs.transpose(0,1))
        # concatenate context vector with last decoder hidden
        rnn_input = torch.cat((motion_input.unsqueeze(1), context), 2)
        
        # now we forward to rest of layers
        rnn_input = self.pre_linear(rnn_input)
        output, hidden = self.gru(rnn_input.transpose(0,1), last_hidden)
        output = self.post_linear(output)
        output = output.squeeze(0)
        
        return output, hidden, attn_weights