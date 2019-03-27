import numpy as np
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer
import torch
from torch.utils import data
import pickle
import re
from tqdm import tqdm

class DataLoader:
    def __init__(self, pickle_file, embedding_file, dimension, components,\
            verbose=False):
        self.pickle_file = pickle_file
        self.embedding_file = embedding_file
        self.dimension = dimension #not the embedding dimension!
        self.verbose = verbose

        self.word2idx = dict()
        self.idx2word = dict()
        self.word_index = {"PAD":0, "SOS":1, "EOS":2}

        vectorizer = CountVectorizer(binary = True, decode_error = u'ignore')
        self.word_tokenizer = vectorizer.build_tokenizer()

        #class flow
        self.text_poses, self.poses = self.loadpickle()
        self.pca = self.runPCA(components)
        self.word_vectors = self.processSentences()

    def loadpickle(self):
        text_poses = []
        poses = []
        """Loads pickle file and loads poses related to the text"""
        with open(self.pickle_file, 'rb') as fp:
            text_pose = pickle.load(fp)
        if self.dimension > len(text_pose):
            tmp_pose = text_pose
        else:
            tmp_pose = text_pose[:self.dimension]
        regex = re.compile('[,\.!?()]')
        text_poses = [[regex.sub('', x[0].lower()), x[1]] for x in tmp_pose]
        poses = [y for x in text_poses for y in x[1] if len(y) > 0]
        """
        for x in tqdm(self.text_poses):
            x[0] = regex.sub('', x[0].lower())
            for y in x[1]:
                if len(y) > 0:
                    self.poses.append(y[:16])
        """
        return text_poses, poses

    def runPCA(self, components):
        pca = PCA(n_components = 10)
        pca.fit(self.poses)
        return pca

    def processSentences(self):
        word_vectors = []
        idx = 0
        with open(self.embedding_file, 'r') as f:
            for line in tqdm(f):
                values = line.split()
                word = values[0]
                word_vectors.append(values[1:])
                self.word2idx[word] = idx
                self.idx2word[idx] = word
                idx += 1
        PAD_TOKEN = 0
        SOS_TOKEN = 1
        EOS_TOKEN = 2
        index = 3
        #dictionary of the unique words in the dataset.
        for x in self.text_poses:
            tmp_y = [y for y in self.word_tokenizer(x[0])]
            for y in tmp_y:
                if y not in self.word_index:
                    self.word_index[y] = index
                    index += 1
        return word_vectors

    def getEmbeddingMatrix(self, emb_dim):
        weight_matrix = np.zeros((len(self.word_index), emb_dim))
        for key, value in self.word_index.items():
            try:
                weight_matrix[value] = self.word_vectors[self.word2idx[key]]
            except KeyError:
                weight_matrix[value] =\
                    np.random.normal(scale=0.6, size=(emb_dim, ))
        return weight_matrix

    def getTrainingData(self):
        x_train = []
        y_train = []
        for x in self.text_poses:
            if len(x[1]) > 0:
                #Processing the embeddings,
                #we transform each string in a sequence of integers, that relate to word indexes.
                tmp = self.word_tokenizer(x[0])
                tmp = [self.word_index[x] for x in tmp]
                #Add start of sentence and end of sentence token.
                tmp.insert(0, self.word_index["SOS"])
                tmp.append(self.word_index["EOS"])
                x_train.append(tmp)
                #restrict first and fourth dimension
                ytemp = self.pca.transform(x[1])
                for z, item in enumerate(ytemp):
                    ytemp[z][1] = -1.00
                    ytemp[z][4] = 1.00
                y_train.append(ytemp)
        return x_train, y_train

class Dataset(data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, x_train, y_train):
        'Initialization'
        self.labels = y_train
        self.list_IDs = x_train
        self.sequence_lengths = [len(x) for x in x_train]
        self.target_lengths = [len(y) for y in y_train]

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        return self.list_IDs[index], self.labels[index],\
            self.sequence_lengths[index], self.target_lengths[index]


def main():
    dl = DataLoader("preprocessed.pickle", "glove.6B.300d.txt", 100, 10)
    print("Loading data...")
    x_train, y_train = dl.getTrainingData()
    print("Making weight matrix...")
    weight_m = dl.getEmbeddingMatrix(300)


if __name__ == '__main__':
    main()
