import numpy as np
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import preprocessing
import torch
from torch.utils import data
import pickle
import re
from tqdm import tqdm

from random import randrange, sample

import matplotlib.pyplot as plt
from matplotlib import pyplot, transforms

class DataLoader:
    def __init__(self, pickle_file, embedding_file, data_size, pca_components=10, verbose=False):
        self.pickle_file = pickle_file
        self.embedding_file = embedding_file
        self.data_size = data_size #not the embedding dimension!
        self.verbose = verbose
        self.pca_components = pca_components

        # here we process text in dataset
        self.wordi2dx = dict()
        self.idx2word = dict()
        self.word_index = {"PAD":0, "SOS":1, "EOS":2}
        vectorizer = CountVectorizer(binary = True, decode_error = u'ignore')
        self.word_tokenizer = vectorizer.build_tokenizer()
        self.word_vectors = self.processSentences()
        
#        get text and pose dataset as json from ted dataset
 #       and run pca through whole poses
        self.ted_data = self.loadpickle(data_size)
        self.pca = self.runPCA(pca_components)
        
    # load texts and poses from ted dataset
    def loadpickle(self, data_size):
        with open(self.pickle_file, 'rb') as fp:
            ted_data = pickle.load(fp)
        if data_size > len(ted_data):
            return ted_data
        else:
            return ted_data[:data_size]

    def get_poses(self):
        poses = []
        for data in self.ted_data:
            if len(data['clips'])>0:
                for clip in data['clips']:      
                    skeletons = clip['skeletons']
                    for sk in skeletons:
                        if sk[0] != 0 and sk[-1] != 0: # remove zero skeltons
                            poses.append(sk)
        # normalize pose data
        poses = np.array(poses)
        normalized_poses = preprocessing.normalize(poses)        

        return normalized_poses
            
    def runPCA(self, components):
        # get all poses in the dataset
        poses = self.get_poses()
        # run pca
        pca = PCA(n_components = 10)
        pca.fit(poses)
        # print("selected dimention: {}".format(pca.n_components_))
        # print("explained variance ratio: {}".format(pca.explained_variance_ratio_))
        return pca

    def processSentences(self):
        word_vectors = []
        idx = 0
        with open(self.embedding_file, 'r') as f:
            for line in tqdm(f):
                values = line.split()
                word = values[0]
                word_vectors.append(values[1:])
                self.wordi2dx[word] = idx
                self.idx2word[idx] = word
                idx += 1
        
        return word_vectors

    def getEmbeddingMatrix(self, emb_dim):
        weight_matrix = np.zeros((len(self.word_index), emb_dim))
        for key, value in self.word_index.items():
            try:
                weight_matrix[value] = self.word_vectors[self.wordi2dx[key]]
            except KeyError:
                weight_matrix[value] = np.random.normal(scale=0.6, size=(emb_dim, ))
                
        return weight_matrix

    def getTrainingData(self):
        x_train = []
        y_train = []
        index = 3 # to create word index

        for data in self.ted_data:
            if len(data['clips'])>0:
                for clip in data['clips']:              
                    
                    # get words in a clip
                    words = clip['words']
                    # assign temp word list
                    sentence_list = []
                    for word in words:
                      w = word[0] # get a word
                      if w != '':
                        sentence_list.append(w) # full sentence
                        # update word index
                        if w not in self.word_index:
                            self.word_index[w] = index
                            index += 1
                    sentence = [self.word_index[w] for w in sentence_list]
                    # add indexed words to x_train
                    x_train.append(sentence)
                    
                    # get skeletons in a clip
                    skeletons = clip['skeletons']
                    pca_skeletons = self.pca.transform(skeletons)
                    for pca_skeleton in pca_skeletons:
                      skeleton_temp = pca_skeleton
                      skeleton_temp[1] = -1.00
                      skeleton_temp[4] = 1.00
                    # add modified skeletons to y_train - todo: need to debug it 
                    y_train.append(pca_skeletons)
        
        tmp_x_list = []
        tmp_y_list = []
        for i in range(len(x_train)):
            if len(x_train[i]) > 10:
                tmp_x_list.append(x_train[i])
                tmp_y_list.append(y_train[i])

        x_train = tmp_x_list
        y_train = tmp_y_list
                    
        return x_train, y_train

    """only used when model infer"""
    def getWordIndex(self, input_string):
        index = 3
        for word in input_string:
            if word not in self.word_index:
                self.word_index[word] = index
                index += 1

class Dataset(data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, x_train, y_train):
        'Initialization'
        self.list_IDs = x_train
        self.labels = y_train
        self.sequence_lengths = [len(x) for x in x_train]
        self.target_lengths = [len(y) for y in y_train]

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        # return self.list_IDs[index], self.labels[index]

        return self.list_IDs[index], self.labels[index], self.sequence_lengths[index], self.target_lengths[index]
    

def plotPose(pose, dl, linewidth=5.0):
    pose = dl.pca.inverse_transform(pose)
    base = pyplot.gca().transData
    rot = transforms.Affine2D().rotate_deg(180)

    plt.plot(pose[0:6:3], pose[1:6:3], transform=rot+base, linewidth=linewidth) #neck
    plt.plot(pose[3:9:3], pose[4:9:3], transform=rot+base,linewidth=linewidth) #sholder 1
    plt.plot([pose[6], pose[9]], [pose[7], pose[10]], transform=rot+base, linewidth=linewidth)
    plt.plot([pose[12], pose[9]], [pose[13], pose[10]], transform=rot+base, linewidth=linewidth) #arm1-1

    plt.plot([pose[3],pose[15]], [pose[4],pose[16]], transform=rot+base, linewidth=linewidth) #sholder 2
    plt.plot([pose[15], pose[18]], [pose[16], pose[19]], transform=rot+base, linewidth=linewidth)
    plt.plot([pose[21], pose[18]], [pose[22], pose[19]], transform=rot+base, linewidth=linewidth) #arm2-1

    plt.show() 

if __name__ == '__main__':
    dataset_size = 10000
    dl = DataLoader("./data/ted_gesture_dataset_train.pickle", "./data/glove.6B.300d.txt", dataset_size, 10)
    dl.getTrainingData()
    # test plot pose 
    mean_pose = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    plotPose(mean_pose, dl)
    