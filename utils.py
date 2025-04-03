import csv
import logging
import math
import matplotlib.pyplot as plt # type: ignore
import numpy as np
import os
import time
import torch
from faiss import IndexFlatL2, IndexPQ, vector_to_array
from pandas import read_csv

import datasets as ds
from bliss_model import BLISS_NN


######################################################################
# Helpers for getting ground truth for train vectors or query vectors.
######################################################################

def get_nearest_neighbours_within_dataset(dataset, amount):
    '''
    Find the true nearest neighbours of vectors within a dataset. To avoid returning a datapoint as its own neighbour, we search for amount+1 neighbours and then filter out
    the first vector (ordered by distance so the vector itself should be the first point).
    '''
    nbrs = IndexFlatL2(dataset.shape[1])
    nbrs.add(dataset)
    _, I = nbrs.search(dataset, amount+1)
    I = I[:, 1:]
    return I

def get_nearest_neighbours_in_different_dataset(dataset, queries, amount):
    '''
    Find the true nearest neighbours of query vectors in dataset. No filtering needed here because the queries do not appear in the dataset.
    '''
    nbrs = IndexFlatL2(dataset.shape[1])
    nbrs.add(dataset)
    _, I = nbrs.search(queries, amount)
    return I

def get_train_nearest_neighbours_from_file(dataset, amount, sample_size, dataset_name, datasize):
    '''
    Helper to read/write nearest neighbour of train data to file so we can test index building without repeating preprocessing each time.
    Should not be used in actual algorithm or experiments where timing the preprocessing is important.
    '''
    if not os.path.exists(f"data/{dataset_name}-nbrs{amount}-sample{sample_size}.csv"):
        filename = f"data/{dataset_name}-nbrs{amount}-sample{sample_size}.csv"
        print(f"no nbrs file found for {dataset_name} with amount={amount} and samplesize={sample_size}, calculating {amount} nearest neighbours")
        logging.info("No neighbours file found, calculating ground truths of training sample")
        I = get_nearest_neighbours_within_dataset(dataset, amount)
        print("writing neighbours to nbrs file")
        I = np.asarray(I)
        with open(filename, "w") as f: 
            np.savetxt(f, I, delimiter=",", fmt='%.0f')

    else:
        print(f"found nbrs file for {dataset_name} with amount={amount} and samplesize={sample_size}, reading true nearest neighbours from file")
        logging.info("Reusing ground truths for training sample from file")
        filename = f"data/{dataset_name}-size{datasize}-nbrs{amount}-sample{sample_size}.csv"
        I = read_csv(filename, dtype=int, header=None).to_numpy()
    return I

######################################################################
# Helpers for training index
######################################################################

def normalise_data(data):
    '''
    Normalize a dataset (divide vectors by their magnitude).
    '''
    norms = np.linalg.norm(data, axis=1, keepdims=True)
    data = data / norms
    return data

def get_training_sample(dataset: ds.Dataset, sample_size, SIZE, DIM):
    if sample_size == SIZE:
        return dataset.get_dataset(), np.arange(0, SIZE)
    sample = np.zeros((sample_size, DIM))
    sample_indexes = np.zeros(sample_size)
    chunk_size = 1_000_000
    chunk_sample_size = sample_size / chunk_size
    index = 0
    for i, batch in enumerate(dataset.get_dataset_iterator(bs=chunk_size)):
        random_order = np.arange(len(batch))
        np.random.seed(i)
        np.random.shuffle(random_order)
        chunk_sample_indexes = np.sort(random_order[:chunk_sample_size])
        sample_indexes[index : index+len(batch)] = chunk_sample_indexes
        sample[index : index+len(batch)] = batch[chunk_sample_indexes]
        index += len(batch)
    return sample, sample_indexes

def get_training_sample_from_memmap(memmap_path, mmp_shape, sample_size, SIZE, DIM):
    '''
    Given a dataset (as a memmap), sample data for model training.
    For small datasets, the full dataset is used to train.
    For large datasets, a random sample is taken across the dataset. It is assumed the sample size is small enough to load the sample into memory.
    '''
    sample = np.zeros(shape=(sample_size, DIM), dtype=np.float32)
    mmp = np.memmap(memmap_path, mode = 'r', shape = mmp_shape, dtype=np.float32)
    if sample_size!=SIZE:
        random_order = np.arange(SIZE)
        np.random.seed(42)
        np.random.shuffle(random_order)
        sample_indexes = np.sort(random_order[:sample_size])
        sample[:] = mmp[sample_indexes, :]
    else:
        sample[:] = mmp
    return sample

def make_ground_truth_labels(B, neighbours, index, sample_size, device):
    '''
    Create ground truth labels for training sample, based on the set of nearest neighbours of each training vector. 
    A label is a B-dimensional vector, where each digit is either 0 (false) if that bucket does not contain any
    nearest neighbours of a vector, and 1 (true) if the bucket contains at least one nearest neighbour of that vector.
    '''
    start = time.time()
    labels = np.zeros((sample_size, B), dtype=bool)
    # for each vector i create an array of amount of neighbours
    vectors = np.concatenate([np.full(len(n), i) for i, n in enumerate(neighbours)])
    # build column indices by applying the mapping to each neighbour array.
    buckets = np.concatenate([index[n] for n in neighbours])
    # Use advanced indexing to set the corresponding entries to True.
    labels[vectors, buckets] = True
    # for i in range(sample_size):
    #     # buckets = index[neighbours[i]]
    #     # labels1[i, buckets] = 1
    #     for neighbour in neighbours[i]:
    #         bucket = index[neighbour]
    #         labels2[i, bucket] = True
    print(f"making ground truch labels {time.time()-start}")
    # if device != torch.device("cpu"):
    #     labels = torch.from_numpy(labels).to(torch.float32)
    return torch.from_numpy(labels)

def reassign_vector_to_bucket(index, bucket_sizes, candidate_buckets, i, item_index):
    '''
    Reassign a vector to the least occupied of the top-k buckets predicted by the model.
    '''
    candidates = candidate_buckets[i]
    candidate_sizes = bucket_sizes[candidates]
    best_bucket = candidates[np.argmin(candidate_sizes)]
    index[item_index] = best_bucket
    bucket_sizes[best_bucket] += 1

def get_all_topk_buckets(loader, k, candidate_buckets, map_model, offset, device):
    logging.info(f"Mapping all train vectors to buckets (baseline)")
    start_idx = offset
    start = time.time()
    with torch.no_grad():
        for batch_data, _, in loader:
            batch_size = len(batch_data)
            batch_candidate_buckets = get_topk_buckets_for_batch(batch_data, k, map_model, device).numpy()
            candidate_buckets[start_idx : start_idx + batch_size, :] = batch_candidate_buckets
            start_idx += batch_size
    print(f"getting top k took {time.time()-start}")

def get_topk_buckets_for_batch(batch_data, k, map_model, device):
    batch_data = batch_data.to(device)
    bucket_probabilities = torch.sigmoid(map_model(batch_data))
    bucket_probabilities_cpu = bucket_probabilities.cpu()
    _, candidate_buckets = torch.topk(bucket_probabilities_cpu, k, dim=1)

    return candidate_buckets

def get_dataset_obj(dataset_name, size):
    '''
    Return a dataset object 
    '''
    if dataset_name == "bigann":
        return ds.BigANNDataset(size)
    elif dataset_name == "deep1b":
        return ds.Deep1BDataset(size)
    elif dataset_name == "sift-128-euclidean":
        return ds.Sift_128()
    elif dataset_name == "glove-100-angular":
        return ds.Glove_100()
    elif dataset_name == "mnist-784-euclidean":
        return ds.Mnist_784()
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

######################################################################
# Helpers for loading and saving models and indexes.
######################################################################

def save_model(model, dataset_name, r, R, K, B, lr, shuffle, reass_mode):
    '''
    Save a (trained) model in the models folder and return the path.
    '''
    model_name = f"model_{dataset_name}_r{r}_k{K}_b{B}_lr{lr}"
    directory = f"models/{dataset_name}_r{R}_k{K}_b{B}_lr{lr}_shf={shuffle}_reass={reass_mode}/"
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

def save_inverted_index(inverted_index, offsets, dataset_name, model_num, R, K, B, lr, shuffle, reass_mode):
    '''
    Save an inverted index (for a specific dataset and parameter setting combination) in the models folder and return the path.
    '''
    index_name = f"index_model{model_num}_{dataset_name}_r{model_num}_k{K}_b{B}_lr{lr}"
    offsets_name = f"offsets_model{model_num}_{dataset_name}_r{model_num}_k{K}_b{B}_lr{lr}"
    directory = f"models/{dataset_name}_r{R}_k{K}_b{B}_lr{lr}_shf={shuffle}_reass={reass_mode}/"
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
    index_path = os.path.join(directory, f"{index_name}.npy")
    offsets_path = os.path.join(directory, f"{offsets_name}.npy")
    np.save(index_path, inverted_index)
    np.save(offsets_path, offsets)
    return index_path

######################################################################
# Helpers for plots created during index building and collecting statistics.
######################################################################

def make_loss_plot(learning_rate, iterations, epochs_per_iteration, k, B, experiment_name, all_losses, shuffle, reass_mode):
    '''
    Plot the total loss of the model after each epoch.
    '''
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
    plt.savefig(f"{foldername}/training_loss_lr={learning_rate}_I={iterations}_E={epochs_per_iteration}_k{k}_B{B}_shf={shuffle}_reass={reass_mode}.png")

def log_mem(function_name, mem_usage, filepath):
    '''
    Log memory usage to a file.
    '''
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
    '''
    Calculate the mean load balance of the bucket assignments of all r indexes for a particular dataset.
    Load balance of a single index is defined as the inverse of the standard deviation of the bucket sizes of that particular index (1-standard deviation).
    Combined load balance for a dataset is then the mean across load balance of r indexes.
    '''
    load_balance_per_model = []
    for r in bucket_size_stats:
        load_balance = 1 / np.std(r)
        load_balance_per_model.append(load_balance)

    avg_load_balance = np.mean(load_balance_per_model)
    return avg_load_balance


######################################################################
# Other helper functions.
######################################################################

def get_best_device():
    '''
    Get the best available torch device (gpu if available, otherwise cpu).
    '''
    if torch.cuda.is_available():
        # covers both NVIDIA CUDA and AMD ROCm
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        # covers apple silicon mps
        return torch.device("mps") 
    else:
        return torch.device("cpu")
    
def set_torch_seed(seed, device):
    '''
    Set torch seed for ease of reproducibility during testing.
    '''
    torch.manual_seed(seed)
    if device == torch.device("cuda"):
        torch.cuda.manual_seed(seed)
    elif device == torch.device("mps"):
        torch.mps.manual_seed(seed)

def train_pq(training_data, m = 8, nbits = 8):
    d = training_data.shape[1]
    pq_index = IndexPQ(d, m , nbits)
    pq_index.train(training_data)
    return (pq_index, m)

def random_projection(X, target_dim):
    original_dim = X.shape[1]
    R = np.random.randn(original_dim, target_dim) / np.sqrt(target_dim)
    return np.dot(X, R)