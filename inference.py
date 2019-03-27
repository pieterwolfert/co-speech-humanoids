import torch
import torch.nn as nn
import numpy as np
from torch import optim
import matplotlib.pyplot as plt

from model import EncoderRNN, AttnDecoderRNN
from dataloader import DataLoader, Dataset
from custom_classes import CustomLoss

__author__ = "Pieter Wolfert"

def speech2gesture(encoder, decoder, input_string):
    """
    Translates an input string into a sequence of co-speech poses.

    Keyword Arguments:
    encoder - trained Encoder
    decoder - trained Decoder
    input_string - integer encoded input string
    """
    encoder.eval()
    decoder.eval()
    #we're going to generate 20 frames per 5 words
    chunks = int(len(input_string) / 5)
    number_of_words = len(input_string)
    rest_words = len(input_string) % 5
    fps_word = 20 / 5
    frames_to_generate = chunks * 20 + rest_words * fps_word
    decoder_fixed_input = torch.\
        FloatTensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    decoder_outputs_generated = torch.zeros(int(frames_to_generate), 1, 10)
    decoder_input = decoder_fixed_input
    for i in range(chunks):
        chunk_input = torch.tensor(input_string[5*i:5*(i+1)]).unsqueeze(0)
        encoder_outputs, encoder_hidden = encoder(chunk_input.transpose(0,1), None)
        decoder_hidden = encoder_hidden[:-2]
        for m in range(20):
            decoder_output, decoder_hidden, attn_weights =\
                decoder(decoder_input, decoder_hidden, encoder_outputs)
            decoder_outputs_generated[(i*20)+m] = decoder_output
            decoder_input = decoder_output
    if rest_words != 0:
        chunk_input = torch.tensor(input_string[-rest_words:]).unsqueeze(0)
        encoder_outputs, encoder_hidden = encoder(chunk_input.transpose(0, 1), None)
        decoder_hidden = encoder_hidden[:-2]
        for n in range(int(rest_words*fps_word)):
            decoder_output, decoder_hidden, attn_weights =\
                decoder(decoder_input, decoder_hidden, encoder_outputs)
            decoder_input = decoder_output
            print((chunks*20)+n)
            decoder_outputs_generated[(chunks*20)+n] = decoder_output
    return decoder_outputs_generated.squeeze(1).detach().numpy()

def loadmodel(model_file, wm, hidden_size, bidirectional):
    """
    Loads the trained model, returns the encoder and decoder for inferencing.
    We initialize 'empty models' in which we will load our parameters.
    It is important that the hyperparameters are the same as used for training.

    Keyword arguments:
    model_file - string with the model location
    wm - embedding matrix
    hidden_size - hidden size
    bidirectional - whether we use bidirectional GRU layers
    """
    model = torch.load(model_file, map_location=lambda storage, loc: storage)
    epoch = model['epoch']
    encoder_state_dict = model['encoder_state_dict']
    encoder_optimizer_state_dict = model['encoder_optimizer_state_dict']
    decoder_state_dict = model['decoder_state_dict']
    decoder_optimizer_state_dict = model['decoder_optimizer_state_dict']
    loss = model['loss']
    encoder = EncoderRNN(wm, 300, hidden_size, bidirectional)
    decoder = AttnDecoderRNN(hidden_size, 10)
    enc_optimizer = optim.Adam(encoder.parameters(), lr=0.0001)
    dec_optimizer = optim.Adam(decoder.parameters(), lr=0.0001)
    return encoder, decoder

def encodeinputstring(input_string, dl):
    """
    We encode our input string the same way as we encoded our training data.

    Keyword Arguments
    input_string - containing what we want to say
    dl - dataloader object that contains our training data
    """
    input_string = [y.lower() for y in dl.word_tokenizer(input_string)]
    tmp = []
    for x in input_string:
        tmp.append(dl.word2idx[x])
    input_string = tmp
    return input_string

def plotPose(pose_list, dl):
    """
    Method for plotting a pose given a list of poses (in pca format).
    The PCA is trained on the training data, and we use the inverse
    transformation to translate the 10 learned factors back to real (normalized)
    coordinates.

    Keyword Arguments
    pose_list - one frame output of the decoder
    dl - dataloader object that contains our training data
    """
    pose_list = dl.pca.inverse_transform(pose_list)
    plt.plot(pose_list[0:10:2], pose_list[1:10:2])
    plt.plot([pose_list[2], pose_list[10]], [pose_list[3], pose_list[11]])
    plt.plot(pose_list[10::2], pose_list[11::2])
    plt.show()

def main():
    #make sure these parameters are the same as in the saved model
    batch_size = 128
    dataset_size = 10000
    hidden_size = 200
    bidirectional = True
    model_file = "./models/seq2seq_90_0.04036942393450761.tar"
    dl = DataLoader("./data/preprocessed_1295videos.pickle",\
        "./data/glove.6B.300d.txt",\
        dataset_size, 10)
    wm = dl.getEmbeddingMatrix(300) #weightmatrix to be used as embedding
    wm = torch.from_numpy(wm).float()

    #load all the prerequisites
    encoder, decoder = loadmodel(model_file, wm, hidden_size, bidirectional)
    input_string = encodeinputstring("this is a very big ball", dl)

    #lets generate poses given the trained model
    to_pose = speech2gesture(encoder, decoder, input_string)

    #let's look at an individual frame:
    plotPose(to_pose[0], dl)

if __name__ == '__main__':
    main()
