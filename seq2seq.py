import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import optim
from torch.utils import data

import pickle
import numpy as np
import random
import time

from tqdm import tqdm
from model import EncoderRNN, AttnDecoderRNN
from dataloader import DataLoader, Dataset
from custom_classes import CustomLoss

import matplotlib.pyplot as plt
import sys

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Seq2Pose():
    def __init__(self, wm, input_length, batch_size, hidden_size, bidirectional, 
                    embedding_size, n_parameter, m_parameter, learning_rate, clip, 
                    alpha, beta, pre_trained_file = None, teacher_forcing_ratio=0.7):
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.bidirectional = bidirectional
        self.n_parameter = n_parameter
        self.m_parameter = m_parameter
        self.learning_rate = learning_rate
        self.wm = wm
        self.clip = clip
        self.alpha = alpha
        self.beta = beta
        self.loss_list = []
        self.teacher_forcing_ratio = teacher_forcing_ratio
        
        if pre_trained_file == None:
            # define encoder and decoder
            self.encoder = EncoderRNN(self.wm, self.embedding_size, hidden_size, bidirectional)
            self.decoder = AttnDecoderRNN("general", self.hidden_size, 10)
            # define optimizer of encoder and decoder
            self.enc_optimizer = optim.Adam(self.encoder.parameters(), lr=self.learning_rate)
            self.dec_optimizer = optim.Adam(self.decoder.parameters(), lr=self.learning_rate)
            self.start = 1
        else:
            self.resume_training = True
            self.encoder, self.decoder, self.enc_optimizer, self.dec_optimizer,\
                self.start = self.load_model_state(pre_trained_file)      
        self.decoder = self.decoder.to(device)
        self.encoder = self.encoder.to(device)


    def load_model_state(self, model_file):
        print("Resuming training from a given model...")
        model = torch.load(model_file, map_location=lambda storage, loc: storage)
        epoch = model['epoch']
        encoder_state_dict = model['encoder_state_dict']
        encoder_optimizer_state_dict = model['encoder_optimizer_state_dict']
        decoder_state_dict = model['decoder_state_dict']
        decoder_optimizer_state_dict = model['decoder_optimizer_state_dict']
        loss = model['loss']
        encoder = EncoderRNN(self.wm, self.embedding_size,\
            self.hidden_size, self.bidirectional)
        decoder = AttnDecoderRNN("general", self.hidden_size, 10)
        enc_optimizer = optim.Adam(encoder.parameters(), lr=self.learning_rate)
        dec_optimizer = optim.Adam(decoder.parameters(), lr=self.learning_rate)
        
        return encoder, decoder, enc_optimizer, dec_optimizer, epoch
    
     
    def train(self, input_batch, target_batch, input_length, target_lengths, criterion):
        # make zero gradient
        self.enc_optimizer.zero_grad()
        self.dec_optimizer.zero_grad()
        
        # fowarding encoder with embedded inputs
        encoder_outputs, encoder_hidden = self.encoder(input_batch, input_length, None)
        decoder_hidden = encoder_hidden[:self.decoder.n_layers]
        
        # variable to store decoder output
        all_decoder_outputs = torch.zeros(target_batch.size(0), target_batch.size(1), target_batch.size(2)).to(device)
        # set initial pose from the selected dataset
        decoder_input = target_batch[0].float()
        all_decoder_outputs[0] = decoder_input
        
        # forwarding decoder with teacher forcing
        # use_teacher_forcing = True if random.random() < self.teacher_forcing_ratio else False # set teacher forcing ratio
        use_teacher_forcing = True
        if use_teacher_forcing:
            for di in range(1, target_lengths):
                decoder_output, decoder_hidden, decoder_attention = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
                all_decoder_outputs[di] = decoder_output
                decoder_input = target_batch[di].float()
        else:
            for di in range(1, target_lengths):
                decoder_output, decoder_hidden, decoder_attention = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
                all_decoder_outputs[di] = decoder_output
                decoder_input = decoder_output.float()
        
        # to calculate continuity of poses, set previous pose (t-1)
        # here we only use 20 poses to calculate loss
        # rest of 10 poses will be used during inferencing
        successive_gesture_generated = all_decoder_outputs[:20]
        decoder_previous_generated = torch.zeros(successive_gesture_generated.size(0),
                                                successive_gesture_generated.size(1),
                                                successive_gesture_generated.size(2)).to(device)
        decoder_previous_generated[1:] = successive_gesture_generated[:-1]
        
        # claculate loss
        loss = criterion(successive_gesture_generated.float(), decoder_previous_generated.float(), target_batch[:20].float())
        loss.backward()
        
        # clip the gradient
        torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), self.clip)
        torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), self.clip)
        # update weigt after one minibatch
        self.enc_optimizer.step()
        self.dec_optimizer.step()
        
        return loss.item()

    def trainIter(self, x_train, y_train, epochs, num_workers=6):
        criterion = CustomLoss(self.alpha, self.beta)
        training_set = Dataset(x_train, y_train)
        training_generator = data.DataLoader(training_set,
                    batch_size=self.batch_size, shuffle=True,
                    collate_fn=self.pad_and_sort_batch,
                    drop_last=True, num_workers=num_workers)
        
        for epoch in range(self.start, epochs+1):
            loss = 0
            for mini_batches in tqdm(training_generator):
                for i, (input_tensor, input_length, output_tensor, output_length) in enumerate(mini_batches):
                    # allocate tensors to GPU
                    input_tensor = input_tensor.to(device)
                    output_tensor = output_tensor.to(device)
                    loss += self.train(input_tensor, output_tensor,
                                        input_length, output_length, criterion)
            print("epoch {} average minibatch loss: {:.6f}".format(epoch, loss/len(training_generator)))

            # save model
            if epoch % 10 == 0:
                self.save_model(self.encoder, self.decoder, self.enc_optimizer,\
                self.dec_optimizer, epoch, "./models/seq2seq_{}_{}.tar".\
                                    format(epoch, loss), loss/len(training_generator))
                print("trained model saved.")
            self.loss_list.append(loss)
            self.loss_graph(self.loss_list) #save loss 
    
    def pad_and_sort_batch(self, DataLoaderBatch):
        batch_size = len(DataLoaderBatch)
        
        # sort by length (descending)
        seq_pairs = sorted(DataLoaderBatch, key=lambda p: len(p[0]), reverse=True)
        batch_split = list(zip(*seq_pairs))
        input_seqs, target_seqs, input_lengths, target_lengths = batch_split[0], batch_split[1], batch_split[2], batch_split[3]   
        
        max_input_length = max(input_lengths) #longest sequence in X
        max_target_length = max(target_lengths) #longest sequence in Y

        # zeropaded to input_seqs
        input_padded = np.zeros((batch_size, max_input_length))
        for i, l in enumerate(input_lengths):
            input_padded[i, 0:l] = input_seqs[i][0:l]
        # zeropaded to target_seqs
        target_padded = np.zeros((batch_size, max([len(s) for s in target_seqs]), 10))
        for t, item in enumerate(target_seqs):
            target_padded[t][:len(target_seqs[t])] = target_seqs[t]

        # make mini batches
        mini_batches = [] #contains x and y tensor per item
        divide = self.m_parameter + self.n_parameter
        #calculating the size for the minibatches
        number_of_chunks = int(max_target_length / (divide))
        
        window_size, overlapped = self.get_sliding_window(number_of_chunks, input_padded.shape[1])
        for nc in range(number_of_chunks):
            # variable to save input length of x
            x_lengths = []
            if nc == 0:
                x = input_padded[:,:window_size]
            else:
                x = input_padded[:,nc*window_size-nc*overlapped:(nc+1)*window_size-nc*overlapped]
            y = target_padded[:, nc*divide:(nc+1)*divide, :]
            # count nonzero and add length of each sequence
            # if there is no length, add 1 (implemented based on https://github.com/pytorch/pytorch/issues/4582)
            for b in range(batch_size):
                length = np.count_nonzero(x[b])
                if length > 0:
                    x_lengths.append(length)
                else:
                    x_lengths.append(1)

            # transpose and make x_train and y_train into torch tensor
            x = np.transpose(x, (1,0))
            input_var = torch.LongTensor(x)
            y = np.transpose(y, (1,0,2))
            target_var = torch.LongTensor(y)
            input_lengths = x_lengths

            mini_batches.append([input_var, input_lengths, target_var, len(y)])
        
        return mini_batches

    def get_sliding_window(self, num_chunk, word_seq_length):
        overlapped_size_alpha = num_chunk - 1
        constant_beta = 4 # we assume that there is 3 overlapped words
        window_size = int((word_seq_length + overlapped_size_alpha*constant_beta) / num_chunk)
        
        return window_size, constant_beta

    def save_model(self, encoder, decoder, enc_optimizer, dec_optimizer,\
        epoch, PATH, loss):
        torch.save({
            'epoch': epoch,
            'encoder_state_dict': encoder.state_dict(),
            'encoder_optimizer_state_dict': enc_optimizer.state_dict(),
            'decoder_state_dict': decoder.state_dict(),
            'decoder_optimizer_state_dict': dec_optimizer.state_dict(),
            'loss': loss,
            }, PATH)

    def loss_graph(self, loss_list, title="loss"):
        plt.plot(loss_list, '-r', label='loss')
        plt.xlabel("epoch")
        plt.title(title)
        plt.savefig("./loss_fig/loss.png")