import numpy as np
import h5py
from sklearn.metrics import pairwise_distances

from utils import random_projection

f = h5py.File("data/sift-128-euclidean.hdf5", 'r')
train = f['train']
X = np.array(train)

target_dim = 64  # Adjust based on the Johnson-Lindenstrauss lemma
projected_X = random_projection(X, target_dim)

original_distances = pairwise_distances(X[:100])
projected_distances = pairwise_distances(projected_X[:100])

mask = original_distances > 0

distortion = np.abs(projected_distances[mask] - original_distances[mask]) / original_distances[mask]
print("average distortion:", np.mean(distortion))
