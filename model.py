import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

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
    def __init__(self, attn_model, hidden_size, output_size, n_layers=1, dropout=0.1):
        super(AttnDecoderRNN, self).__init__()
        
        self.attn_model = attn_model
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout = dropout
        
        # self.attn = Attn(attn_model, hidden_size)
        self.attn = Attn_B(hidden_size)
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


class Attn_B(nn.Module):
    def __init__(self, hidden_size):
        super(Attn_B, self).__init__()
        self.hidden_size = hidden_size
        self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
        self.v = nn.Parameter(torch.rand(hidden_size))
        stdv = 1. / math.sqrt(self.v.size(0))
        self.v.data.normal_(mean=0, std=stdv)

    def forward(self, hidden, encoder_outputs):
        '''
        :param hidden:
            previous hidden state of the decoder, in shape (layers*directions,B,H)
        :param encoder_outputs:
            encoder outputs from Encoder, in shape (T,B,H)
        :return
            attention energies in shape (B,T)
        '''
        max_len = encoder_outputs.size(0)
        this_batch_size = encoder_outputs.size(1)
        H = hidden.repeat(max_len, 1, 1).transpose(0, 1)
        encoder_outputs = encoder_outputs.transpose(0, 1)  # [B*T*H]
        attn_energies = self.score(H, encoder_outputs)  # compute attention score
        return F.softmax(attn_energies, dim=1).unsqueeze(1)  # normalize with softmax

    def score(self, hidden, encoder_outputs):
        energy = torch.tanh(self.attn(torch.cat([hidden, encoder_outputs], 2)))  # [B*T*2H]->[B*T*H]
        energy = energy.transpose(2, 1)  # [B*H*T]
        v = self.v.repeat(encoder_outputs.data.shape[0], 1).unsqueeze(1)  # [B*1*H]
        energy = torch.bmm(v, energy)  # [B*1*T]
        return energy.squeeze(1)  # [B*T]


class BahdanauAttnDecoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers=1, dropout_p=0.1,
                 discrete_representation=False):
        super(BahdanauAttnDecoderRNN, self).__init__()

        # define parameters
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p
        self.discrete_representation = discrete_representation

        # define embedding layer
        if self.discrete_representation:
            self.embedding = nn.Embedding(output_size, hidden_size)
            self.dropout = nn.Dropout(dropout_p)

        # calc input size
        if self.discrete_representation:
            input_size = hidden_size  # embedding size
        linear_input_size = input_size + hidden_size

        # define layers
        self.pre_linear = nn.Sequential(
            nn.Linear(linear_input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True)
        )
        self.attn = Attn_B(hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=dropout_p)

        # self.out = nn.Linear(hidden_size * 2, output_size)
        self.out = nn.Linear(hidden_size, output_size)

    def freeze_attn(self):
        for param in self.attn.parameters():
            param.requires_grad = False

    def forward(self, motion_input, last_hidden, encoder_outputs):
        '''
        :param motion_input:
            motion input for current time step, in shape [batch x dim]
        :param last_hidden:
            last hidden state of the decoder, in shape [layers x batch x hidden_size]
        :param encoder_outputs:
            encoder outputs in shape [steps x batch x hidden_size]
        :return
            decoder output
        Note: we run this one step at a time i.e. you should use a outer loop
            to process the whole sequence
        '''

        if '1.2.0' in torch.__version__:
            self.gru.flatten_parameters()

        if self.discrete_representation:
            word_embedded = self.embedding(motion_input).view(1, motion_input.size(0), -1)  # [1 x B x embedding_dim]
            motion_input = self.dropout(word_embedded)
        else:
            motion_input = motion_input.view(1, motion_input.size(0), -1)  # [1 x batch x dim]

        # attention
        attn_weights = self.attn(last_hidden[-1], encoder_outputs)  # [batch x 1 x T]
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))  # [batch x 1 x attn_size]
        context = context.transpose(0, 1)  # [1 x batch x attn_size]

        # make input vec
        rnn_input = torch.cat((motion_input, context), 2)  # [1 x batch x (dim + attn_size)]

        rnn_input = self.pre_linear(rnn_input.squeeze(0))
        rnn_input = rnn_input.unsqueeze(0)

        # rnn
        output, hidden = self.gru(rnn_input, last_hidden)

        # post-fc
        output = output.squeeze(0)  # [1 x batch x hidden_size] -> [batch x hidden_size]
        output = self.out(output)

        return output, hidden, attn_weights