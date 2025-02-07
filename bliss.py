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

data = []
labels = []

class Dataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # turn nd.array into tensor when fetched from the Dataset
        vector = torch.from_numpy(self.data[idx])
        label = self.data[idx]
        return vector, label

dataset = Dataset(data, labels)

class BLISS_NN(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(DIMENSION, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, B),
        )

    def forward(self, vector):
        logits = self.linear_relu_stack(vector)
        return logits
    
def train():
    model = BLISS_NN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    num_epochs = 5

    for i in range(20):
        for epoch in range(num_epochs):
            for batch_data, batch_labels in dataloader:
                # Convert numpy arrays to torch tensors
                batch_data = torch.tensor(batch_data)
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
    train, _ = generate_random_array(SIZE, dimensions=DIMENSION, centers=1)
    index, counts = assign_initital_buckets(len(train), 1, B)
    neighbours = get_nearest_neighbors(train, 10)
    labels = make_ground_truth_labels(B, neighbours=neighbours, index=index)
    print(neighbours)
    print(labels)
    dataset = Dataset(train, labels)
    # for vector, labels in dataset:
    #     print("vector__________________________")
    #     print(vector)
    #     print("labels__________________________")
    #     print(labels)
