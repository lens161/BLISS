from sklearn.neighbors import NearestNeighbors
from datasets import *
import torch
import faiss 
import os
import h5py
import numpy as np
import traceback
import math
import csv
import pickle
import matplotlib.pyplot as plt # type: ignore
from sklearn.model_selection import train_test_split as sklearn_train_test_split
from urllib.request import Request, urlopen
from pandas import read_csv

def get_nearest_neighbours_old(train, amount):
    nbrs = NearestNeighbors(n_neighbors=amount+1, metric="euclidean", algorithm='brute').fit(train)
    I = nbrs.kneighbors(train, return_distance=False)
    I = I[:, 1:]
    return I

def get_nearest_neighbours_old_without_filter(data, queries, amount):
    nbrs = NearestNeighbors(n_neighbors=amount, metric="euclidean", algorithm='brute').fit(data)
    I = nbrs.kneighbors(queries, return_distance=False)
    return I

def get_nearest_neighbours_faiss_within_dataset(dataset, amount):
    '''
    Find the true nearest neighbours of vectors within a dataset. To avoid returning a datapoint as its own neighbour, we search for amount+1 neighbours and then filter out
    the first vector (ordered by distance so the vector itself should be the first point).
    '''
    nbrs = faiss.IndexFlatL2(dataset.shape[1])
    nbrs.add(dataset)
    _, I = nbrs.search(dataset, amount+1)
    I = I[:, 1:]
    return I

def get_nearest_neighbours_faiss_in_different_dataset(dataset, queries, amount):
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
    Should not be used in actual algorithm or experiments.
    '''
    if not os.path.exists(f"data/{dataset_name}-nbrs{amount}-sample{sample_size}.csv"):
        print(f"no nbrs file found for {dataset_name} with amount={amount} and samplesize={sample_size}, calculating {amount} nearest neighbours")
        I = get_nearest_neighbours_faiss_within_dataset(dataset, amount)
        print("writing neighbours to nbrs file")
        I = np.asarray(I)
        np.savetxt(f"data/{dataset_name}-nbrs{amount}-sample{sample_size}.csv", I, delimiter=",", fmt='%.0f')
    else:
        print(f"found nbrs file for {dataset_name} with amount={amount} and samplesize={sample_size}, reading true nearest neighbours from file")
        filename = f"data/{dataset_name}-nbrs{amount}-sample{sample_size}.csv"
        I = read_csv(filename, dtype=int, header=None).to_numpy()
    return I

def get_dataset_obj(dataset_name, size):
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

# def read_dataset(dataset_name, size = 1):
#     if not os.path.exists("data"):
#         os.mkdir("data")
    
#     mmp_path = f"memmaps/memmap_{dataset_name}_{size}.npy" if size > 1 else f"memmaps/memmap_{dataset_name}.npy"
#     print(f"loading {dataset_name}...")
#     dataset = get_dataset(dataset_name, size)
#     dataset.prepare()
#     print(f"dataset size = {dataset.nb_M}")
#     if not os.path.exists(mmp_path):
#         data = dataset.get_dataset()
#         mmap = np.lib.format.open_memmap(mmp_path, mode='w+', shape=data.shape, dtype=data.dtype)
#         print(f"saving {dataset_name} to memmap...")
#         mmap[:] = data[:]
#     return dataset

def get_B(n):
    if n > 0:
        sq = math.sqrt(n)
        B = 2 ** round(math.log(sq, 2))
        return B
    else:
        raise Exception(f"cannot calculate B for empty dataset!")
    
def save_model(model, dataset_name, r, R, K, B, lr, shuffle, global_reass):
    model_name = f"model_{dataset_name}_r{r}_k{K}_b{B}_lr{lr}"
    directory = f"models/{dataset_name}_r{R}_k{K}_b{B}_lr{lr}_shf={shuffle}_gr={global_reass}/"
    MODEL_PATH = os.path.join(directory, f"{model_name}.pt")
    
    os.makedirs(directory, exist_ok=True)
    
    torch.save(model.state_dict(), MODEL_PATH)
    return MODEL_PATH

def save_inverted_index(inverted_index, dataset_name, model_num, R, K, B, lr, shuffle, global_reass):
    index_name = f"index_model{model_num}_{dataset_name}_r{model_num}_k{K}_b{B}_lr{lr}"
    directory = f"models/{dataset_name}_r{R}_k{K}_b{B}_lr{lr}_shf={shuffle}_gr={global_reass}/"
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
    index_path = os.path.join(directory, f"{index_name}.pkl")
    with open(index_path, 'wb') as f:
        pickle.dump(inverted_index, f)
    return index_path

def load_indexes(dataset_name, R, K):
    indexes = []
    if not os.path.exists(f"models/{dataset_name}_{R}_{K}/"): 
        print("no index found for this data set or configuration: build index first")
    else:
         for i in range(R):
             indexes.append(np.load(f"models/{dataset_name}_{R}_{K}/model_{dataset_name}_r{i}_k{K}.pt"))
    return indexes

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

## function below is old version of save_dataset_as_memmap in case new one fucks something up
# def save_dataset_as_memmap(train, rest, dataset_name, train_on_full_dataset):
#     memmap_name = f"memmap_{dataset_name}"
#     memmap_path = f"memmaps/{memmap_name}.npy"
#     if not os.path.exists("memmaps/"):
#         os.mkdir("memmaps")
#     size_train, dim = np.shape(train)
#     size_rest = 0
#     if not train_on_full_dataset:
#         size_rest, _ = np.shape(rest)
#     size = size_rest + size_train
#     print(f"size = {size}")
 
#     memmap = np.memmap(memmap_path, dtype=float, mode='w+', shape=(size, dim))
#     if size_rest == 0:
#         memmap[:] = train[:]
#     else:
#         all = np.concatenate((train, rest), axis=0)
#         memmap[:] = all[:]
#     # np.append(memmap, rest)
#     memmap.flush()
#     return memmap

def save_dataset_as_memmap(data, dataset_name):
    dir_path = "memmaps/"
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
    file_path = os.path.join(dir_path, f"memmap_{dataset_name}.npy")

    # size_train, dim = np.shape(train)
    # size_rest = 0
    # if not train_on_full_dataset:
    #     size_rest, _ = np.shape(rest)
    # size = size_train + size_rest
    # print(f"Total size = {size}")

    # combine train and rest if needed
    # if size_rest == 0:
    # combined = train
    # else:
    #     combined = np.concatenate((train, rest), axis=0)
    
    np.save(file_path, data)
    print(f"Dataset saved to {file_path} with shape {data.shape}")

def get_best_device():
    if torch.cuda.is_available():
        # covers both NVIDIA CUDA and AMD ROCm
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        # apple silicon mps
        return torch.device("mps") 
    else:
        return torch.device("cpu")
    
def split_training_sample(data, sample_size):
    '''
    seperate training sample from data
    '''
    print(f"Splitting training sample from {data}")
    return sklearn_train_test_split(data, test_size=sample_size, random_state=1)