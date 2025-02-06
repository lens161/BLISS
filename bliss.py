import numpy as np
import torch
from sklearn.utils import murmurhash3_32 as mmh3

def assign_buckets(train_size, r, B):
    index = np.zeros(train_size)
    counts = np.zeros(B)

    for i in range(train_size):
        bucket = mmh3(i,seed=r)%B
        index[i] = bucket
        counts[bucket] += 1
    
    return index, counts

# make ground truth labels
# for each vector:
#   initialize label of 0's
#   look up bucket of each NN in index
#   if bucket is 0 in label, change to 1
#   put label somewhere

if __name__ == "__main__":
    index, counts = assign_buckets(1000000, 4, 1024)
    print(index)