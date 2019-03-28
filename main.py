from seq2seq import Seq2Pose
from dataloader import DataLoader
import torch

__author__ = "Pieter Wolfert"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def main():
    batch_size = 128
    dataset_size = 10000
    hidden_size = 200
    bidirectional = True
    embedding_size = 300
    epochs = 500
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
        learning_rate, clip, alpha, beta, pre_trained_file = "./models/seq2seq_10_0.13292566299438477.tar")
    sq.train(epochs, x_train, y_train)

if __name__=="__main__":
    main()
