from sklearn.neighbors import NearestNeighbors
import torch
import faiss 
import os
import h5py
import numpy as np
import traceback
import math
from sklearn.model_selection import train_test_split as sklearn_train_test_split
from urllib.request import Request, urlopen
from pandas import read_csv

def get_nearest_neighbours_old(train, amount):
    nbrs = NearestNeighbors(n_neighbors=amount+1, metric="euclidean", algorithm='brute').fit(train)
    I = nbrs.kneighbors(train, return_distance=False)
    I = I[:, 1:]
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

def get_train_nearest_neighbours_from_file(dataset, amount, dataset_name):
    '''
    Helper to read/write nearest neighbour of train data to file so we can test index building without repeating preprocessing each time.
    Should not be used in actual algorithm or experiments.
    '''
    if not os.path.exists(f"data/{dataset_name}-trainnbrs-{amount}.csv"):
        print(f"no nbrs file found for {dataset_name} with amount={amount}, calculating {amount} nearest neighbours")
        I = get_nearest_neighbours_faiss_within_dataset(dataset, amount)
        print("writing neighbours to nbrs file")
        I = np.asarray(I)
        np.savetxt(f"data/{dataset_name}-trainnbrs-{amount}.csv", I, delimiter=",", fmt='%.0f')
    else:
        print(f"found nbrs file for {dataset_name} with amount={amount}, reading true nearest neighbours from file")
        filename = f"data/{dataset_name}-trainnbrs-{amount}.csv"
        I = read_csv(filename, dtype=int, header=None).to_numpy()
    return I

def read_dataset(dataset_name):
    if not os.path.exists("data"):
        os.mkdir("data")
    path = os.path.join("data", f"{dataset_name}.hdf5")

    url = f"https://ann-benchmarks.com/{dataset_name}.hdf5"

    if not os.path.exists(path):
        try:
            # Add custom headers to bypass 403 error
            req = Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            print(f"downloading dataset {dataset_name}...")
            with urlopen(req) as response, open(path, 'wb') as out_file:
                out_file.write(response.read())
        except:
            traceback.print_exc()
            raise Exception(f"dataset: {dataset_name} could not be downloaded")

    print(f"reading file: {path}")
    f = h5py.File(path)
    train = f['train']
    test = f['test']
    neighbours = f['neighbors']
    train_X = np.array(train)
    test_X = np.array(test)
    neighbours_X = np.array(neighbours)
    print("done reading")
    return train_X, test_X, neighbours_X

def get_B(n):
    if n > 0:
        sq = math.sqrt(n)
        B = 2 ** round(math.log(sq, 2))
        return B
    else:
        raise Exception(f"cannot calculate B for empty dataset!")
    
def save_model(model, dataset_name, R, K):
    model_name = f"model_{dataset_name}_{R}_{K}"
    MODEL_PATH = f"models/{model_name}.pt"
    if not os.path.exists("models/"):
        os.mkdir("models")
    torch.save(model.state_dict(), MODEL_PATH)
    return MODEL_PATH

def save_dataset_as_memmap(train, rest, dataset_name):
    memmap_name = f"memmap_{dataset_name}"
    memmap_path = f"memmaps/{memmap_name}.npy"
    if not os.path.exists("memmaps/"):
        os.mkdir("memmaps")
    size_train, dim = np.shape(train)
    size_rest, _ = np.shape(rest)
    size = size_rest + size_train
    print(f"size = {size}")

    memmap = np.memmap(memmap_path, dtype=float, mode='w+', shape=(size, dim))
    all = np.concatenate((train, rest), axis=0)
    # np.append(memmap, rest)
    memmap[:] = all[:]
    memmap.flush()
    return memmap

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