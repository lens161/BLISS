import numpy as np
import torch
import os
import pandas as pd
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from sklearn.utils import murmurhash3_32 as mmh3
from utils import *

device = get_best_device()
print("Using device:", device)
DIMENSION = 128
B = 1024

data = []
labels = []

class Dataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        vector = self.data[idx]
        label = self.data[idx]
        return vector, label

dataset = Dataset(data, labels)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=2)

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

def assign_buckets(train_size, r, B):
    index = np.zeros(train_size, dtype=int) # from 0 to train_size-1
    counts = np.zeros(B)

    for i in range(train_size):
        bucket = mmh3(i,seed=r)%B
        index[i] = bucket
        counts[bucket] += 1
    
    return index, counts

def make_ground_truth_labels(B, neighbours, index):
    size = len(index)
    labels = np.zeros((B, size), dtype=bool)

    for i in range(size):
        for neighbour in neighbours[i]:
            bucket = index[neighbour]
            labels[bucket, i] = True
    
    return labels

if __name__ == "__main__":
    index, counts = assign_buckets(1000000, 4, 1024)
    print(index)