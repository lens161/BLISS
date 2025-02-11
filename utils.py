from sklearn.neighbors import NearestNeighbors
import torch
import faiss 
import os
import h5py
import numpy as np
import traceback
import math
from urllib.request import urlretrieve, Request, urlopen

def get_nearest_neighbours(train, amount):
    nbrs = NearestNeighbors(n_neighbors=amount+1, metric="euclidean", algorithm='brute').fit(train)
    I = nbrs.kneighbors(train, return_distance=False)
    I = I[:, 1:]
    return I

def get_nearest_neighbours_faiss(train, amount):
    nbrs = faiss.IndexFlatL2(train.shape[1])
    nbrs.add(train)
    _, I = nbrs.search(train, amount+1)
    I = I[:, 1:]
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

def get_B(b):
    sq = math.sqrt(b)
    B = 2 ** round(math.log(sq, 2))
    return B

def get_best_device():
    if torch.cuda.is_available():
        # covers both NVIDIA CUDA and AMD ROCm
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        # apple silicon mps
        return torch.device("mps")
    else:
        return torch.device("cpu")