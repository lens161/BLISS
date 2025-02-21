import numpy as np
import matplotlib.pyplot as plt
import sys
import time
import torch
import statistics
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
# from torchvision import datasets, transforms
from sklearn.utils import murmurhash3_32 as mmh3
from utils import *

class BLISSDataset(Dataset):
    def __init__(self, data, labels, device):
        self.device = device
        self.labels = labels
        if device == torch.device("cpu"):
            self.data = data
        else:
            self.data = torch.from_numpy(data).float()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # turn nd.array into tensor when fetched from the Dataset
        if self.device == torch.device("cpu"):
            vector = torch.from_numpy(self.data[idx]).float()
            label = torch.from_numpy(self.labels[idx]).float()
        else:
            vector = self.data[idx]
            label = self.labels[idx]
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

def train_model(model, dataset, index, iterations, k, B, sample_size, bucket_sizes, neighbours, epochs_per_iteration, batch_size, device):
    model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    learning_rate = 0.001
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    all_losses = []

    for i in range(iterations):
        model.train() 
        for epoch in range(epochs_per_iteration):
            epoch_losses = []
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
                epoch_losses.append(loss.item())
            finish = time.time()
            elapsed = finish-start
            print(f"epoch {epoch} took {elapsed}")
            avg_loss = statistics.mean(epoch_losses)
            print(f"epoch {epoch} avg. loss = {avg_loss}")
            all_losses.append(avg_loss)
        reassign_buckets(model, dataset, k, B, index, bucket_sizes, sample_size, neighbours, batch_size, device)
        print(f"index after iteration {i} = {index}")

    plt.figure(figsize=(10, 5))
    plt.plot(all_losses, marker='o')
    plt.title('Training Loss Over Epochs')
    plt.xlabel('Epoch (accumulated over iterations)')
    plt.ylabel('Average Loss')
    plt.grid(True)

    plt.savefig(f"training_loss_lr={learning_rate}_I={iterations}_E={epochs_per_iteration}.png")
    plt.show()

def reassign_buckets(model, dataset, k, B, index, bucket_sizes, sample_size, neighbours, batch_size, device):
    sample_size, _ = np.shape(dataset.data)
    model.to("cpu")
    model.eval()
    reassign_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    bucket_sizes[:] = 0
    item_index = 0

    start = time.time()
    with torch.no_grad():
        for batch_data, batch_labels in reassign_loader:
            batch_data = batch_data.to("cpu")
            bucket_probabilities = torch.sigmoid(model(batch_data))

            for probability_vector in bucket_probabilities:
                reassign_vector_to_bucket(probability_vector, index, bucket_sizes, k, item_index)
                item_index += 1
                     
    finish = time.time()
    elapsed = finish - start
    print(f"reassigning took {elapsed}")
    new_labels = make_ground_truth_labels(B, neighbours, index, sample_size, device)
    dataset.labels = new_labels
    model.to(device)

def reassign_vector_to_bucket(probability_vector, index, bucket_sizes, k, item_index):
    value, indices_of_topk_buckets = torch.topk(probability_vector, k)
    smallest_bucket = indices_of_topk_buckets[0]
    smallest_bucket_size = bucket_sizes[smallest_bucket]
    for i in indices_of_topk_buckets:
        size = bucket_sizes[i]
        if size < smallest_bucket_size:
            smallest_bucket = i
            smallest_bucket_size = size

    index[item_index] = smallest_bucket
    bucket_sizes[smallest_bucket] +=1  
            
def assign_initial_buckets(train_size, rest_size, r, B):
    '''
    assign bucket labels to vectors (indeces in the nd.array) using a hash function.
    the hash fucntion used here is the same as in the original code from the BLISS github.
    TO-DO: add reference link
    '''
    index = np.zeros(train_size+rest_size, dtype=np.uint32) # from 0 to train_size-1
    bucket_sizes = np.zeros(B, dtype=np.uint32)

    for i in range(train_size):
        bucket = mmh3(i,seed=r)%B
        index[i] = bucket
        bucket_sizes[bucket] += 1
    
    return index, bucket_sizes

def make_ground_truth_labels(B, neighbours, index, sample_size, device):
    # size = sample_size
    labels = np.zeros((sample_size, B), dtype=bool)

    for i in range(sample_size):
        for neighbour in neighbours[i]:
            bucket = index[neighbour]
            labels[i, bucket] = True
    if device != torch.device("cpu"):
        labels = torch.from_numpy(labels).float()
    return labels

def map_all_to_buckets(rst_vectors, k, bucket_sizes, index, model_path, training_sample_size, DIMENSION, B):
    rst_vectors = torch.from_numpy(rst_vectors)
    print(f"training sample size = {training_sample_size}")
    map_model = BLISS_NN(DIMENSION, B)
    map_model.load_state_dict(torch.load(model_path, weights_only=True))
    map_model.eval()

    for i, vector in enumerate(rst_vectors, start=training_sample_size):
        if i < training_sample_size:
            print("wrong start")
        scores = map_model(vector)
        probabilities = torch.sigmoid(scores)
        values, candidates = torch.topk(probabilities, k)
        smallest_bucket = candidates[0]
        smallest_bucket_size = bucket_sizes[smallest_bucket]
        # print(candidates)
        for cand in candidates:
            size = bucket_sizes[cand]
            if size < smallest_bucket_size:
                smallest_bucket = cand
                smallest_bucket_size = size
        
        index[i] = smallest_bucket
        bucket_sizes[smallest_bucket] +=1


def invert_index(index, B):
    inverted_index = [[] for _ in range(B)]
    for i, bucket in enumerate(index):
        inverted_index[bucket].append(i)
    return inverted_index

def get_sample(train, SIZE, DIMENSION):
    sample_size = SIZE if SIZE < 10_000_000 else int(0.01*SIZE)
    print(f"sample size = {sample_size}")
    sample = np.empty((sample_size, DIMENSION))
    # rest = np.empty((int((1-sample_size_percentage)*SIZE), DIMENSION))
    rest = None
    rest_size = 0
    train_on_full_dataset = (sample_size == SIZE)
    if not train_on_full_dataset:
        sample, rest = split_training_sample(train, SIZE-sample_size)
        rest_size, _ = np.shape(rest)
    else:
        sample = train
    
    return sample, rest, sample_size, rest_size, train_on_full_dataset

def build_index(BATCH_SIZE, EPOCHS, ITERATIONS, R, K, NR_NEIGHBOURS, device, dataset_name):
    train, _, _ = read_dataset(dataset_name)
    print("training data_________________________")
    print(f"train shape = {np.shape(train)}")

    SIZE, DIMENSION = np.shape(train)
    B = get_B(SIZE)
    print(f"nr of buckets (B): {B}")

    sample, rest, sample_size, rest_size, train_on_full_dataset = get_sample(train, SIZE, DIMENSION)

    print(f"writing train vectors to memmap")
    memmap = save_dataset_as_memmap(sample, rest, dataset_name, train_on_full_dataset)
    print(f"memmap = {memmap}")
    print(f"memmap shape = {np.shape(memmap)}")

    print("looking for true neighbours of training sample")
    neighbours = get_train_nearest_neighbours_from_file(sample, NR_NEIGHBOURS, sample_size, dataset_name)
    print(neighbours)
    labels = []
    dataset = BLISSDataset(sample, labels, device)

    final_index = []
    # build R models/indexes
    for r in range(R):

        print(f"randomly assigning initial buckets")
        index, bucket_sizes = assign_initial_buckets(sample_size, rest_size, r, B)
        print(bucket_sizes)
        print("making initial ground truth labels")
        labels = make_ground_truth_labels(B, neighbours, index, sample_size, device)
        dataset.labels = labels # replace old labels in dataset with new labels for current model 
        print(f"setting up model {r+1}")
        model = BLISS_NN(DIMENSION, B)
        print(f"training model {r+1}")
        train_model(model, dataset, index, ITERATIONS, K, B, sample_size, bucket_sizes, neighbours, EPOCHS, BATCH_SIZE, device)
        model_path = save_model(model, dataset_name, r+1, R, K)
        print(f"model {r+1} saved to {model_path}")
        print(f"model {r+1}: index before full assignment = {index}")
        # np.set_printoptions(threshold=np.inf, suppress=True)
        # print(f"model {R}: buckets before full assignment = {bucket_sizes}")
        np.set_printoptions(threshold=6, suppress=True)
        if not train_on_full_dataset:
            print("assigning rest of vectors to buckets")
            map_all_to_buckets(rest, K, bucket_sizes, index, model_path, sample_size, DIMENSION, B)
        print(f"model {r+1}: index after full assignment{index}")
        np.set_printoptions(threshold=np.inf, suppress=True)
        print(bucket_sizes)
        inverted_index = invert_index(index, B)
        index_path = save_inverted_index(inverted_index, dataset_name, r+1, R, K)
        np.set_printoptions(threshold=1000, suppress=True)
        final_index.append((index_path, model_path))
    # return paths to all models created for the index
    return final_index

def load_model(model_path, dim, b):
    inf_device = torch.device("cpu")
    model = BLISS_NN(dim, b)
    model.load_state_dict(torch.load(model_path, weights_only=True, map_location=inf_device))
    model.eval()
    return model

def query_multiple(data, index, vectors, m, threshold, requested_amount):
    '''run multiple queries from a set of query vectors i.e. "Test" from the ANN benchmark datsets'''
    size = len(vectors)
    results = [[] for i in range(size)]
    for i in range(size):
        print(f"querying {i} of {size}")
        vector = vectors[i]
        anns = query(data, index, vector, m, threshold, requested_amount)
        results[i] = anns
    return results

def query(data, index, query_vector, m, freq_threshold, requested_amount):
    '''query the index for a single vector'''
    inverted_indexes, models = index
    buckets = set()
    candidates = {}
    for i in range(len(models)):
        model = models[i]
        index = inverted_indexes[i]
        probabilities = model(query_vector)
        _, m_buckets = torch.topk(probabilities, m)
        for bucket in m_buckets:
            # print(f"bucket {bucket} size = {len(index[bucket])}")
            for vector in index[bucket]:
                if candidates.__contains__(vector):
                    f = candidates.get(vector)
                else:
                    f = 0
                candidates.update({vector : f+1})
    final_results = [key for key, value in candidates.items() if value >= freq_threshold]
    # print (f"final results = {len(final_results)}")
    if len(candidates) <= requested_amount:
        return np.array(candidates)
    else:
        return reorder(data, query_vector, np.array(final_results, dtype=int), requested_amount)

def reorder(data, query_vector, candidates, requested_amount):
    import faiss 
    #  TO-DO: get this shit to work....
    n, d = np.shape(data)
    sp_index = {}
    search_space = data[candidates]
    for i in range(len(search_space)):
        sp_index.update({i : candidates[i]})
    print(f"search_space = {search_space}")
    index = faiss.IndexFlatL2(d)
    index.add(search_space)
    query_vector = query_vector.reshape(1, -1)
    (dist, neighbours) = index.search(query_vector, requested_amount)
    neighbours = neighbours[0].tolist()
    final_neighbours = []
    for neighbour in neighbours:
        final_neighbours.append(sp_index.get(neighbour))
    return final_neighbours

def recall(results, neighbours):
    recalls = []
    for ann, nn in zip(results, neighbours):
        # get size of intersection of anns and nns to find the amount of correct anns
        correct = len(set(ann) & set(nn))
        recalls.append(correct/len(nn))
    return np.mean(recalls)

def recall_single(results, neighbours):
    return len(set(results) & set(neighbours))/len(neighbours)

if __name__ == "__main__":
    BATCH_SIZE = 256
    EPOCHS = 5   
    ITERATIONS = 20
    R = 1
    K = 2
    NR_NEIGHBOURS = 100
    device = get_best_device()
    # device = "cpu"
    print("Using device:", device) 

    dataset_name = "sift-128-euclidean"
    # dataset_name = "mnist-784-euclidean"

    # index = build_index(BATCH_SIZE, EPOCHS, ITERATIONS, R, K, NR_NEIGHBOURS, device, dataset_name)
    # inverted_indexes_paths, model_paths = zip(*index)

    inverted_indexes_paths = [f"models/sift-128-euclidean_4_2/index_model{i+1}_sift-128-euclidean_r{i+1}_k2.pkl" for i in range(R)]
    for i in range (R):
        inverted_indexes_paths.append(f"models/sift-128-euclidean_4_2/index_model{i+1}_sift-128-euclidean_r{i+1}_k2.pkl")
    model_paths = [f"models/sift-128-euclidean_4_2/model_sift-128-euclidean_r{i+1}_k2.pt" for i in range(R)]
    for i in range(R):
        model_paths.append(f"models/sift-128-euclidean_4_2/model_sift-128-euclidean_r{i+1}_k2.pt")

    inverted_indexes = []
    for path in inverted_indexes_paths:
        with open(path, 'rb') as f:
            inverted_indexes.append(pickle.load(f))

    # print(f"inv indexes = {inverted_indexes}")
    q_models = [load_model(model_path, 128, 1024) for model_path in model_paths]
    index = (inverted_indexes, q_models)

    data, test, neighbours = read_dataset(dataset_name)

    print(f"neigbours = {len(neighbours[0])}")

    print(f"ctreating np array from Test")
    test = torch.from_numpy(test)

    # results = query_multiple(data, index, test, 2, 3, 100)
    results = query(data, index, test[0], 2, 2, 100)
    print(f"results = {results}")

    RECALL = recall_single(results, neighbours[0])
    print(f"RECALL = {RECALL}")