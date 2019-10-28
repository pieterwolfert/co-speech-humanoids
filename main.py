from seq2seq import Seq2Pose
from dataloader import DataLoader
import torch
import os

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True
torch.multiprocessing.set_sharing_strategy('file_system')

def main():
    batch_size = 128
    dataset_size = 10000

    # batch_size = 3
    # dataset_size = 3

    hidden_size = 200
    bidirectional = True
    embedding_size = 300
    epochs = 500
    learning_rate = 0.0001
    clip = 5.0
    alpha = 0.1
    beta = 1
    
    # n and m are parameters from the paper
    n = 10
    m = 20
    
    dl = DataLoader("./data/ted_gesture_dataset_train.pickle",\
        "./data/glove.6B.300d.txt", dataset_size, 10)
    x_train, y_train = dl.getTrainingData()
    print("paris of sequence of words and poses: {}".format(len(x_train)))

    # we make sure the embedding is a torch tensor
    wm = dl.getEmbeddingMatrix(300)
    wm = torch.from_numpy(wm).float().to(device)
    sq = Seq2Pose(wm, 24, batch_size, hidden_size, True, embedding_size, n , m, 
                    learning_rate, clip, alpha, beta, decoder_type="original")
                    # pre_trained_file="./models/seq2seq_2_4619870.238758475.tar")
    sq.trainIter(x_train, y_train, epochs, num_workers=6)

if __name__=="__main__":
    main()
