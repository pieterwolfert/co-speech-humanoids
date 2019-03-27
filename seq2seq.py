import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import numpy as np
import random
import time

from torch import optim
from torch.utils import data
from torch.autograd import Variable
from tqdm import tqdm

from model import EncoderRNN, AttnDecoderRNN
from dataloader import DataLoader, Dataset
from custom_classes import CustomLoss

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Seq2Pose():
    def __init__(self, wm, input_length, batch_size, hidden_size, bidirectional\
            , embedding_size, n_parameter, m_parameter, learning_rate, clip,\
                alpha, beta):
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.n_parameter = n_parameter
        self.m_parameter = m_parameter
        self.learning_rate = learning_rate
        self.clip = clip
        self.alpha = alpha
        self.beta = beta
        self.encoder = EncoderRNN(wm, embedding_size, hidden_size, bidirectional)
        self.encoder = self.encoder.to(device)
        self.decoder = AttnDecoderRNN(hidden_size, 10)
        self.decoder = self.decoder.to(device)

    def train(self, epochs, x_train, y_train):
        """
        Training loop, trains the network for the given parameters.

        Keyword arguments:
        epochs - number of epochs to train for (looping over the whole dataset)
        x_train - training data, contains a list of integer encoded strings
        y_train - training data, contains a list of pose sequences
        """
        enc_optimizer = optim.Adam(self.encoder.parameters(),\
            lr=self.learning_rate)
        dec_optimizer = optim.Adam(self.decoder.parameters(),\
            lr=self.learning_rate)
        criterion = CustomLoss(self.alpha, self.beta)
        training_set = Dataset(x_train, y_train)
        training_generator = data.DataLoader(training_set,\
            batch_size=self.batch_size, shuffle=True,\
            collate_fn=self.pad_and_sort_batch,\
            num_workers=8, drop_last=True)

        decoder_fixed_previous = Variable(torch.zeros(self.n_parameter,\
            self.batch_size, 10, requires_grad=False)).to(device)
        decoder_fixed_input = torch.FloatTensor\
            ([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]] *\
                self.batch_size).to(device)

        for epoch in range(epochs):
            total_loss = 0
            for mini_batches, max_target_length in tqdm(training_generator):
                #kickstart vectors
                enc_optimizer.zero_grad()
                dec_optimizer.zero_grad()
                loss = 0
                decoder_previous_inputs = decoder_fixed_previous
                for z in range(self.n_parameter):
                    decoder_previous_inputs[z] = decoder_fixed_input
                for i, (x, y, lengths) in enumerate(mini_batches):
                    t1 = time.perf_counter()
                    x = x.to(device)
                    y = y.to(device)
                    decoder_m = np.shape(y)[0]
                    encoder_outputs, encoder_hidden = self.encoder(x, None)
                    decoder_hidden = encoder_hidden[:self.decoder.n_layers]
                    decoder_output = None
                    for n_prev in range(self.n_parameter):
                        decoder_output, decoder_hidden, attn_weights =\
                            self.decoder(decoder_previous_inputs[n_prev].float(),\
                                decoder_hidden, encoder_outputs)
                    decoder_input = decoder_output.float()
                    decoder_previous_generated = Variable(torch.zeros(decoder_m,\
                        self.batch_size, 10, requires_grad=False)).to(device)
                    decoder_outputs_generated = Variable(torch.zeros(decoder_m,\
                        self.batch_size, 10, requires_grad=False)).to(device)
                    for fut_pose in range(decoder_m):
                        decoder_output, decoder_hidden, attn_weights =\
                            self.decoder(decoder_input,decoder_hidden, encoder_outputs)
                        decoder_outputs_generated[fut_pose] = decoder_output
                        decoder_input = y[fut_pose].float()
                    decoder_previous_inputs = decoder_outputs_generated[:-10]
                    # max_length, batch_, item
                    # now mask generated outputs
                    decoder_masked = torch.where(y == 0.0, y.float(),\
                        decoder_outputs_generated.float())
                    decoder_previous_generated[1:] = decoder_masked[:-1]
                    loss += criterion(decoder_masked, decoder_previous_generated,\
                        y.float())
                    total_loss += loss.item()

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.encoder.parameters(),\
                        self.clip)
                torch.nn.utils.clip_grad_norm_(self.decoder.parameters(),\
                        self.clip)
                enc_optimizer.step()
                dec_optimizer.step()

            if epoch % 10 == 0:
                self.save_model(self.encoder, self.decoder, enc_optimizer,\
                    dec_optimizer, epoch, "./models/seq2seq_{}_{}.tar".\
                    format(epoch, total_loss/len(x_train)), total_loss)
            print("Epoch: {} Loss: {}".format(epoch, total_loss))


    def pad_and_sort_batch(self, DataLoaderBatch):
        """
        Pads and sorts the batches, provided as a collate function.

        Keyword arguments:
        DataLoaderBatch - Batch of data coming from dataloader class.
        """
        batch_size = len(DataLoaderBatch)
        batch_split = list(zip(*DataLoaderBatch))

        seqs, targs, lengths, target_lengths = batch_split[0], batch_split[1],\
            batch_split[2], batch_split[3]

        #calculating the size for the minibatches
        max_length = max(lengths) #longest sequence in X
        max_target_length = max(target_lengths) #longest sequence in Y
        number_of_chunks = int(max_target_length / self.m_parameter)
        not_in_chunk = max_target_length % self.m_parameter
        words_per_chunk = int(max_length / number_of_chunks)
        not_in_words_per_chunk = max_length % words_per_chunk

        #first zeropad it all
        padded_seqs = np.zeros((batch_size, max_length))
        for i, l in enumerate(lengths):
            padded_seqs[i, 0:l] = seqs[i][0:l]
        new_targets = np.zeros((batch_size, max([len(s) for s in targs]), 10))
        for i, item in enumerate(targs):
            new_targets[i][:len(targs[i])] = targs[i]
        seq_lengths, perm_idx = torch.tensor(lengths).sort(descending=True)
        seq_lengths = list(seq_lengths)
        seq_tensor = padded_seqs[perm_idx]
        target_tensor = new_targets[perm_idx]
        #Full batch is sorted, now we are going to create minibatches.
        #in these batches time comes first, so: [time, batch, features]
        #we also add a vector with lengths, which are necessary for padding
        mini_batches = [] #contains x and y tensor per item
        seq_tensor = np.transpose(seq_tensor, (1,0))
        target_tensor = np.transpose(target_tensor, (1,0,2))
        counter = 0
        for i in range(number_of_chunks):
            x = seq_tensor[i*words_per_chunk:(i+1)*words_per_chunk]
            y = target_tensor[i*self.m_parameter:(i+1)*self.m_parameter]
            counter += words_per_chunk*i
            x_mini_batch_lengths = []
            for j in range(batch_size):
                if seq_lengths[j] > counter and seq_lengths[j] < counter + words_per_chunk:
                    x_mini_batch_lengths.append(seq_lengths[j].item() - counter)
                elif seq_lengths[j] > counter + words_per_chunk:
                    x_mini_batch_lengths.append(words_per_chunk)
                else:
                    x_mini_batch_lengths.append(0)
            mini_batches.append([torch.tensor(x).long(), torch.tensor(y), x_mini_batch_lengths])
        if not_in_chunk != 0:
            x = seq_tensor[number_of_chunks*words_per_chunk:]
            y = target_tensor[number_of_chunks*self.m_parameter:]
            x_mini_batch_lengths = []
            counter = number_of_chunks * words_per_chunk
            for j in range(batch_size):
                if seq_lengths[j] > counter and seq_lengths[j] < counter + words_per_chunk:
                    x_mini_batch_lengths.append(seq_lengths[j].item() - counter)
                elif seq_lengths[j] > counter + words_per_chunk:
                    x_mini_batch_lengths.append(words_per_chunk)
                else:
                    x_mini_batch_lengths.append(0)
            if len(x) > 0 and len(y) > 0:
                mini_batches.append([torch.tensor(x).long(), torch.tensor(y), x_mini_batch_lengths])
        return mini_batches, max_target_length


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

def main():
    batch_size = 64
    dataset_size = 100
    hidden_size = 200
    bidirectional = True
    embedding_size = 300
    epochs = 10
    learning_rate = 0.0001
    clip = 5.0
    alpha = 0.1
    beta = 1
    #n and m are parameters from the paper
    n = 10
    m = 20

    #dataloader loads our preprocessed pickle file, and the pretrained embedding.
    dl = DataLoader("./data/preprocessed_1295videos.pickle",\
        "./data/glove.6B.300d.txt", dataset_size, 10)
    x_train, y_train = dl.getTrainingData()

    #we make sure the embedding is a torch tensor
    wm = dl.getEmbeddingMatrix(300)
    wm = torch.from_numpy(wm).float().to(device)
    sq = Seq2Pose(wm, 24, batch_size, hidden_size, True, embedding_size, n , m,\
        learning_rate, clip, alpha, beta)
    sq.train(epochs, x_train, y_train)

if __name__=="__main__":
    main()
