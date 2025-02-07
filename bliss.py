import numpy as np
import math 
import torch
import os
import pandas as pd
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from sklearn.utils import murmurhash3_32 as mmh3
from utils import *
from test import *

device = get_best_device()
print("Using device:", device)
SIZE = 10000
DIMENSION = 128
B = int(math.sqrt(SIZE))
BATCH_SIZE = 32
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
        label = self.labels[idx]
        return vector, label

dataset = Dataset(train, labels)
vector, label = dataset.__getitem__(0)

class BLISS_NN(nn.Module):
    def __init__(self, input_size, output_size):
        super(BLISS_NN, self).__init__()
        # takes input and projects it to 512 hidden neurons
        self.fc1 = nn.Linear(input_size, 512)
        # activation function
        self.relu = nn.ReLU()
        # output layer maps 512 hidden neurons to output neurons (representing the buckets)
        self.fc2 = nn.Linear(512, output_size)
        # turns all output values into softmax values that sum to 1 -> probabilities
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # x is  training vector?
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)

        output = torch.softmax(x, dim=1)
        return output

def train(model, dataset, iterations, epochs_per_iteration=EPOCHS):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    for i in range(iterations):
        for epoch in range(epochs_per_iteration):
            for batch_data, batch_labels in dataloader:
                # Convert numpy arrays to torch tensors
                batch_labels = torch.tensor(batch_labels)
                
                # Zero the parameter gradients
                optimizer.zero_grad()
                
                # Forward pass
                outputs = model(batch_data)
                loss = criterion(outputs, batch_labels)
                
                # Backward pass and optimization
                loss.backward()
                optimizer.step()
                
        # TO-DO:
        # reassignment of labels after 5 epochs

def reassign_buckets(model, dataset, batch_size = BATCH_SIZE):
    model.eval()
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

def assign_initital_buckets(train_size, r, B):
    '''
    assign bucket labels to vectors (indeces in the nd.array) using a hash function.
    the hash fucntion used here is the same as in the original code from the BLISS github.
    TO-DO: add reference link
    '''
    index = np.zeros(train_size, dtype=int) # from 0 to train_size-1
    counts = np.zeros(B)

    for i in range(train_size):
        bucket = mmh3(i,seed=r)%B
        index[i] = bucket
        counts[bucket] += 1
    
    return index, counts

def make_ground_truth_labels(B, neighbours, index):
    size = len(index)
    labels = np.zeros((size, B), dtype=bool)

    for i in range(size):
        for neighbour in neighbours[i]:
            bucket = index[neighbour]
            labels[i, bucket] = True
    
    return labels

if __name__ == "__main__":
    SIZE = 1000
    train, test = generate_random_array(size=SIZE, dimensions=DIMENSION, centers=1)
    index, counts = assign_initital_buckets(len(train), 1, B)
    neighbours = get_nearest_neighbors(train, 10)
    labels = make_ground_truth_labels(B=B, neighbours=neighbours, index=index)
    print(neighbours)
    print(labels)
    dataset = Dataset(train, labels)
    # for vector, labels in dataset:
    #     print("vector__________________________")
    #     print(vector)
    #     print("labels__________________________")
    #     print(labels)
