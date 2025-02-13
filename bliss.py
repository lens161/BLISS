import numpy as np
import time
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
# from torchvision import datasets, transforms
from sklearn.utils import murmurhash3_32 as mmh3
from utils import *

class BLISSDataset(Dataset):
    def __init__(self, data, labels):
        # Convert the whole dataset to tensors once
        self.data = torch.from_numpy(data).float()
        self.labels = labels

    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

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

def train_model(model, dataset, index, iterations, k, bucket_sizes, neighbours, epochs_per_iteration, batch_size, SIZE):
    model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=6)

    for i in range(iterations):
        model.train()
        for epoch in range(epochs_per_iteration):
            print(f"training epoch ({i}, {epoch})")
            start = time.time()
            for batch_data, batch_labels in train_loader:
                batch_data = batch_data.to(device)
                batch_labels = batch_labels.to(device)
                # print(f"batch_labels: {batch_labels}")
                # Zero the parameter gradients
                optimizer.zero_grad()
                # Forward pass
                probabilities = model(batch_data)
                loss = criterion(probabilities, batch_labels)
                # Backward pass and optimization
                loss.backward()
                optimizer.step()
            finish = time.time()
            elapsed = finish-start
            print(f"epoch {epoch} took {elapsed}")
        reassign_buckets(model, dataset, k, index, bucket_sizes, neighbours, batch_size, SIZE)
        print(f"index after iteration {i} = {index}")

def reassign_buckets(model, dataset, k, index, bucket_sizes, neighbours, batch_size, SIZE):
    model.to("cpu")
    model.eval()
    reassign_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=6)
    item_index = 0

    start = time.time()
    with torch.no_grad():
        for batch_data, batch_labels in reassign_loader:
            batch_data = batch_data.to("cpu")
            bucket_probabilities = torch.sigmoid(model(batch_data))
            # print(f"bucket probabilities: {bucket_probabilities}")

            for probability_vector in bucket_probabilities:
                reassign_vector_to_bucket(probability_vector, index, SIZE, bucket_sizes, k, item_index)
                item_index += 1
                     
    finish = time.time()
    elapsed = finish - start
    print(f"reassigning took {elapsed}")
    new_labels = make_ground_truth_labels(B, neighbours, index)
    dataset.labels = new_labels
    model.to(device)

def reassign_vector_to_bucket(probability_vector, index, SIZE, bucket_sizes, k, item_index):
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
    labels = torch.from_numpy(labels).float()
    return labels

if __name__ == "__main__":
    BATCH_SIZE = 256
    EPOCHS = 5
    ITERATIONS = 20
    K = 2
    NR_NEIGHBOURS = 100
    device = get_best_device()
    # device = "cpu"
    print("Using device:", device)

    dataset_name = "sift-128-euclidean"
    # dataset_name = "mnist-784-euclidean"
    train, _, _ = read_dataset(dataset_name)
    print("training data_________________________")
    print(np.shape(train))

    SIZE, DIMENSION = np.shape(train)
    B = get_B(SIZE)
    print(B)

    R = 1
    index, counts = assign_initital_buckets(len(train), R, B)

    print("looking for true neighbours")
    neighbours = get_train_nearest_neighbours_from_file(train, NR_NEIGHBOURS, dataset_name)
    print(neighbours)

    print("making ground truth labels")
    labels = make_ground_truth_labels(B, neighbours, index)

    dataset = BLISSDataset(train, labels)
    model = BLISS_NN(DIMENSION, B)

    print("training model")
    train_model(model, dataset, index, ITERATIONS, K, counts, neighbours, EPOCHS, BATCH_SIZE, SIZE)

    np.set_printoptions(threshold=np.inf, suppress=True)
    print(counts)

