from sklearn.neighbors import NearestNeighbors
import torch

def get_nearest_neighbors(train, amount):
    nbrs = NearestNeighbors(n_neighbors=amount, metric="euclidean", algorithm='brute').fit(train)
    I = nbrs.kneighbors(train, return_distance=False)
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