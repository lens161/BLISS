import numpy as np
import torch
import os
# import pandas as pd
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from sklearn.utils import murmurhash3_32 as mmh3
import sklearn.datasets
from sklearn.model_selection import train_test_split as sklearn_train_test_split
from sklearn.neighbors import NearestNeighbors
from utils import *
import sys
import math
# import importlib
# importlib.reload(utils) 
device = get_best_device()
# device = "cpu"
print("Using device:", device)

# SIZE = 0
# DIMENSION = 0
# B = 0
BATCH_SIZE = 256
EPOCHS = 5

class Dataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # turn nd.array into tensor when fetched from the Dataset
        vector = torch.from_numpy(self.data[idx]).float()
        label = torch.from_numpy(self.labels[idx]).float()
        return vector, label

class BLISS_NN(nn.Module):
    def __init__(self, input_size, output_size):
        super(BLISS_NN, self).__init__()
        # takes input and projects it to 512 hidden neurons
        # fc stands for fully connected, referring to a fully connected matrix being created
        self.fc1 = nn.Linear(input_size, 512)
        # activation function
        self.relu = nn.ReLU()
        # output layer maps 512 hidden neurons to output neurons (representing the buckets)
        self.fc2 = nn.Linear(512, output_size)
        # turns all output values into softmax values that sum to 1 -> probabilities
        # self.sigmoid = nn.Sigmoid(dim=1)

    def forward(self, x):
        # x is  training vector?
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)

        return x

def train_model(model, dataset, index, iterations, k, bucket_sizes, neighbours, epochs_per_iteration=EPOCHS):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    for i in range(iterations):
        model.train()
        for epoch in range(epochs_per_iteration):
            print(f"training epoch ({i}, {epoch})")
            for batch_data, batch_labels in train_loader:
                # print(f"batch_labels: {batch_labels}")
                # Zero the parameter gradients
                optimizer.zero_grad()
                # Forward pass
                probabilities = model(batch_data)
                loss = criterion(probabilities, batch_labels)
                # Backward pass and optimization
                loss.backward()
                optimizer.step()
        reassign_buckets(model, dataset, k, index, bucket_sizes, neighbours, batch_size=BATCH_SIZE)
        print(f"index after iteration {i} = {index}")

        # TO-DO:
        # reassignment of labels after 5 epochs

def reassign_buckets(model, dataset, k, index, bucket_sizes, neighbours, batch_size = BATCH_SIZE):
    model.eval()
    reassign_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    item_index = 0

    for batch_data, batch_labels in reassign_loader:
        bucket_probabilities = torch.sigmoid(model(batch_data))
        # print(f"bucket probabilities: {bucket_probabilities}")

        for probability_vector in bucket_probabilities:
            value, indices_of_topk_buckets = torch.topk(probability_vector, k)
            # print(f"indices = {indices_of_topk_buckets}")
            smallest_bucket = 0
            smallest_bucket_size = SIZE
            old_bucket = index[item_index]
            for i in indices_of_topk_buckets:
                size = bucket_sizes[i]
                if size < smallest_bucket_size:
                    smallest_bucket = i
                    smallest_bucket_size = size
        
            index[item_index] = smallest_bucket
            bucket_sizes[old_bucket] -=1
            bucket_sizes[smallest_bucket] +=1
            item_index+=1        
    new_labels = make_ground_truth_labels(B, neighbours, index)
    dataset.labels = new_labels
    
            
def assign_initital_buckets(train_size, r, B):
    '''
    assign bucket labels to vectors (indeces in the nd.array) using a hash function.
    the hash fucntion used here is the same as in the original code from the BLISS github.
    TO-DO: add reference link
    '''
    index = np.zeros(train_size, dtype=int) # from 0 to train_size-1
    bucket_sizes = np.zeros(B)

    for i in range(train_size):
        bucket = mmh3(i,seed=r)%B
        index[i] = bucket
        bucket_sizes[bucket] += 1
    
    return index, bucket_sizes

def make_ground_truth_labels(B, neighbours, index):
    size = len(index)
    labels = np.zeros((size, B), dtype=bool)

    for i in range(size):
        for neighbour in neighbours[i]:
            bucket = index[neighbour]
            labels[i, bucket] = True
    
    return labels

if __name__ == "__main__":
    # train, _, _ = read_dataset("mnist-784-euclidean")
    train, _, _ = read_dataset("sift-128-euclidean")
    print("training data_________________________")
    print(np.shape(train))

    SIZE, DIMENSION = np.shape(train)
    B = get_B(SIZE)
    # B = 128
    print(B)
    BATCH_SIZE = 256

    index, counts = assign_initital_buckets(len(train), 1, B)

    print("looking for true neighbours")
    neighbours = get_nearest_neighbours_faiss(train, 100)

    print("making ground truth labels")
    labels = make_ground_truth_labels(B, neighbours=neighbours, index=index)

    dataset = Dataset(train, labels)
    model = BLISS_NN(DIMENSION, B)

    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model.to(device)
    print("training model")
    train_model(model, dataset, index, 5, 2, counts, neighbours, 5)
    np.set_printoptions(threshold=np.inf, suppress=True)
    # print(index)
    print(counts)
