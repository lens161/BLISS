from sklearn.neighbors import NearestNeighbors
import torch
import faiss 

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

def get_best_device():
    if torch.cuda.is_available():
        # covers both NVIDIA CUDA and AMD ROCm
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        # apple silicon mps
        return torch.device("mps")
    else:
        return torch.device("cpu")