import numpy as np
import torch
import sklearn.datasets
from sklearn.model_selection import train_test_split as sklearn_train_test_split
from sklearn.neighbors import NearestNeighbors
import h5py
from utils import *
from bliss import *

def generate_random_array(size: int, dimensions: int, centers: int):
    '''
    generate random training and test arrays
    '''
    X, _ = sklearn.datasets.make_blobs(n_samples=size, n_features=dimensions, centers=centers, random_state=1)
    train, test = train_test_split(X, test_size=0.1)
    return train, test

def train_test_split(X, test_size):
    '''
    split an nd.array into training and test data
    '''
    # dimension = dimension if not None else X.shape[1]
    print(f"Splitting into train/test")
    return sklearn_train_test_split(X, test_size=test_size, random_state=1)

def make_train_test_tensors(train, test):
    t_train = []
    t_test = []
    for vector in train:
        t_train.append(torch.from_numpy(vector))
    for vector in test:
        t_test.append(torch.from_numpy(vector))
    return t_train, t_test


if __name__ == "__main__":
    dimension = 128
    train, test = generate_random_array(100, dimension, 1)
    index, _ = assign_initital_buckets(len(train), 1, B)
    t_train, t_test = make_train_test_tensors(train, test)
    print(t_train)
    print(t_test)
    train_NN = get_nearest_neighbors(train, 5)
    train_pairs = zip(t_train, train_NN)
    nr_buckets = 5
    index, counts = assign_initital_buckets(len(train), 2, nr_buckets)
    labels = make_ground_truth_labels(nr_buckets, train_NN, index)
    print(labels)

