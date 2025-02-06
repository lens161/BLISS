import numpy as np
import torch
from sklearn.utils import murmurhash3_32 as mmh3
from utils import *

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