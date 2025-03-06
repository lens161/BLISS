import numpy as np
# import random
import sklearn.datasets
import unittest
import matplotlib.pyplot as plt # type: ignore
# import pandas as pd
import pickle
import matplotlib.colors as mcolors
# from sklearn.model_selection import train_test_split as sklearn_train_test_split
from utils import *
from bliss import *

class TestTrainMethods(unittest.TestCase):
  
    def test_synthetic_data_heavily_clustered(self):
        b = 32
        k = 2
        r = 1
        nr_neighbours = 3
        batch_size = 256
        epochs = 5
        iterations = 20
        device = torch.device('cpu')
        lr = 0.001
        experiment_name = "test"
        dataset_name = "synthetic_data"
        data, _ = sklearn.datasets.make_blobs(n_samples=1000, n_features=2, centers=5, random_state=1, cluster_std=0.5)
        data = data.astype('float32')
        # print(np.shape(data))
        # print(data)

        index, time_per_r, build_time, memory_usage = build_index(batch_size, epochs, iterations, r, k, nr_neighbours, device, data, dataset_name, b, lr, experiment_name)
        inverted_indexes_paths, model_paths = zip(*index)
        # print(inverted_indexes_paths[0])
        inverted_index = None
        with open(inverted_indexes_paths[0], 'rb') as f:
            inverted_index = pickle.load(f)
        
        # print(inverted_index)
        # print(mcolors.XKCD_COLORS)
        # colourlist = mcolors.XKCD_COLORS.keys()

        plt.figure(figsize = (8,5))
        for i in range(b):
            # print(inverted_index[i])
            relevant_vectors = [data[j] for j in inverted_index[i]]
            # print(relevant_vectors)
            flipped = np.transpose(relevant_vectors)
            # print(flipped)
            if len(flipped) > 0:
                plt.scatter(flipped[0], flipped[1], c=list(mcolors.XKCD_COLORS.values())[i], s=1)
        
        plt.show()


if __name__ == "__main__":
    unittest.main()