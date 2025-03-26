# from sklearn.neighbors import NearestNeighbors
from datasets import *
import torch
import faiss 
import os
import numpy as np
import math
import csv
import pickle
import matplotlib.pyplot as plt # type: ignore
from sklearn.model_selection import train_test_split as sklearn_train_test_split
from pandas import read_csv
from bliss_model import BLISS_NN

'''
Helpers for getting ground truth for train vectors or query vectors.
'''

def get_nearest_neighbours_within_dataset(dataset, amount):
    '''
    Find the true nearest neighbours of vectors within a dataset. To avoid returning a datapoint as its own neighbour, we search for amount+1 neighbours and then filter out
    the first vector (ordered by distance so the vector itself should be the first point).
    '''
    nbrs = faiss.IndexFlatL2(dataset.shape[1])
    nbrs.add(dataset)
    _, I = nbrs.search(dataset, amount+1)
    I = I[:, 1:]
    return I

def get_nearest_neighbours_in_different_dataset(dataset, queries, amount):
    '''
    Find the true nearest neighbours of query vectors in dataset. No filtering needed here because the queries do not appear in the dataset.
    '''
    nbrs = faiss.IndexFlatL2(dataset.shape[1])
    nbrs.add(dataset)
    _, I = nbrs.search(queries, amount)
    return I

def get_train_nearest_neighbours_from_file(dataset, amount, sample_size, dataset_name):
    '''
    Helper to read/write nearest neighbour of train data to file so we can test index building without repeating preprocessing each time.
    Should not be used in actual algorithm or experiments where timing the preprocessing is important.
    '''
    if not os.path.exists(f"data/{dataset_name}-nbrs{amount}-sample{sample_size}.csv"):
        print(f"no nbrs file found for {dataset_name} with amount={amount} and samplesize={sample_size}, calculating {amount} nearest neighbours")
        I = get_nearest_neighbours_within_dataset(dataset, amount)
        print("writing neighbours to nbrs file")
        I = np.asarray(I)
        np.savetxt(f"data/{dataset_name}-nbrs{amount}-sample{sample_size}.csv", I, delimiter=",", fmt='%.0f')
    else:
        print(f"found nbrs file for {dataset_name} with amount={amount} and samplesize={sample_size}, reading true nearest neighbours from file")
        filename = f"data/{dataset_name}-nbrs{amount}-sample{sample_size}.csv"
        I = read_csv(filename, dtype=int, header=None).to_numpy()
    return I

'''
Helpers for training index
'''

def make_ground_truth_labels(B, neighbours, index, sample_size, device):
    '''
    Create ground truth labels for training sample, based on the set of nearest neighbours of each training vector. 
    A label is a B-dimensional vector, where each digit is either 0 (false) if that bucket does not contain any
    nearest neighbours of a vector, and 1 (true) if the bucket contains at least one nearest neighbour of that vector.
    '''
    labels = np.zeros((sample_size, B), dtype=bool)
    for i in range(sample_size):
        for neighbour in neighbours[i]:
            bucket = index[neighbour]
            labels[i, bucket] = True
    if device != torch.device("cpu"):
        labels = torch.from_numpy(labels).float()
    return labels

def reassign_vector_to_bucket(probability_vector, index, bucket_sizes, k, item_index):
    '''
    Reassign a vector to the least occupied of the top-k buckets predicted by the model.
    '''
    value, indices_of_topk_buckets = torch.topk(probability_vector, k)
    # get sizes of candidate buckets
    candidate_sizes = bucket_sizes[indices_of_topk_buckets]
    # get bucket at index of smallest bucket from bucket_sizes
    best_bucket = indices_of_topk_buckets[np.argmin(candidate_sizes)]
    index[item_index] = best_bucket
    bucket_sizes[best_bucket] +=1  

def get_dataset_obj(dataset_name, size):
    '''
    Return a dataset object 
    '''
    if dataset_name == "bigann":
        return BigANNDataset(size)
    elif dataset_name == "deep1b":
        return Deep1BDataset(size)
    elif dataset_name == "sift-128-euclidean":
        return Sift_128()
    elif dataset_name == "glove-100-angular":
        return Glove_100()
    else:
        print("dataset not supported yet")

def get_B(n):
    '''
    Calculated suggested B (nr of buckets) based on the size of the dataset. Recommended B is the first power of 2 larger than sqrt(n).
    '''
    if n > 0:
        sq = math.sqrt(n)
        B = 2 ** round(math.log(sq, 2))
        return B
    else:
        raise Exception(f"cannot calculate B for empty dataset!")

'''
Helpers for loading and saving models and indexes.
'''

def save_model(model, dataset_name, r, R, K, B, lr, shuffle, global_reass):
    '''
    Save a (trained) model in the models folder and return the path.
    '''
    model_name = f"model_{dataset_name}_r{r}_k{K}_b{B}_lr{lr}"
    directory = f"models/{dataset_name}_r{R}_k{K}_b{B}_lr{lr}_shf={shuffle}_gr={global_reass}/"
    MODEL_PATH = os.path.join(directory, f"{model_name}.pt")
    
    os.makedirs(directory, exist_ok=True)
    
    torch.save(model.state_dict(), MODEL_PATH)
    return MODEL_PATH

def load_model(model_path, dim, b):
    '''
    Load a (trained) model from the specified path for inference (weights only).
    '''
    inf_device = torch.device("cpu")
    model = BLISS_NN(dim, b)
    model.load_state_dict(torch.load(model_path, weights_only=True, map_location=inf_device))
    model.eval()
    return model

def save_inverted_index(inverted_index, dataset_name, model_num, R, K, B, lr, shuffle, global_reass):
    '''
    Save an inverted index (for a specific dataset and parameter setting combination) in the models folder and return the path.
    '''
    index_name = f"index_model{model_num}_{dataset_name}_r{model_num}_k{K}_b{B}_lr{lr}"
    directory = f"models/{dataset_name}_r{R}_k{K}_b{B}_lr{lr}_shf={shuffle}_gr={global_reass}/"
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
    index_path = os.path.join(directory, f"{index_name}.pkl")
    with open(index_path, 'wb') as f:
        pickle.dump(inverted_index, f)
    return index_path

# def load_indexes(dataset_name, R, K):
#     '''
#     Load all R inverted indexes for a specific dataset and parameter setting combination.
#     '''
#     indexes = []
#     if not os.path.exists(f"models/{dataset_name}_{R}_{K}/"): 
#         print("no index found for this data set or configuration: build index first")
#     else:
#          for i in range(R):
#              indexes.append(np.load(f"models/{dataset_name}_{R}_{K}/model_{dataset_name}_r{i}_k{K}.pt"))
#     return indexes

# def save_dataset_as_memmap(data, dataset_name):
#     '''
    
#     '''
#     dir_path = "memmaps/"
#     if not os.path.exists(dir_path):
#         os.mkdir(dir_path)
#     file_path = os.path.join(dir_path, f"memmap_{dataset_name}.npy")

#     np.save(file_path, data)
#     print(f"Dataset saved to {file_path} with shape {data.shape}")

'''
Helpers for plots created during index building and collecting statistics.
'''

def make_loss_plot(learning_rate, iterations, epochs_per_iteration, k, B, experiment_name, all_losses, shuffle, global_reass):
    foldername = f"results/{experiment_name}"
    if not os.path.exists("results"):
        os.mkdir("results")
    if not os.path.exists(foldername):
        os.mkdir(foldername)
    plt.figure(figsize=(10, 5))
    plt.plot(all_losses, marker='.')
    plt.title('Training Loss Over Epochs')
    plt.xlabel('Epoch (accumulated over iterations)')
    plt.ylabel('Average Loss')
    plt.grid(True)
    plt.savefig(f"{foldername}/training_loss_lr={learning_rate}_I={iterations}_E={epochs_per_iteration}_k{k}_B{B}_shf={shuffle}_gr={global_reass}.png")

def log_mem(function_name, mem_usage, filepath):
    file_exists = os.path.isfile(filepath)
    with open(filepath, mode='a', newline='') as csv_file:
        fieldnames = ['function', 'memory_usage_mb']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            'function': function_name,
            'memory_usage_mb': mem_usage
        })

def calc_load_balance(bucket_size_stats):
    load_balance_per_model = []
    for r in bucket_size_stats:
        load_balance = 1 / np.std(r)
        load_balance_per_model.append(load_balance)

    avg_load_balance = np.mean(load_balance_per_model)
    return avg_load_balance


'''
Other helper functions.
'''

def get_best_device():
    if torch.cuda.is_available():
        # covers both NVIDIA CUDA and AMD ROCm
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        # covers apple silicon mps
        return torch.device("mps") 
    else:
        return torch.device("cpu")
