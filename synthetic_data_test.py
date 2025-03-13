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
<<<<<<< HEAD
        N = 1000
        blobs = 1
        R = 4
        d = 2
        b = 32
        k = 2
        r = 1
        nr_neighbours = 3
        batch_size = 256
        epochs = 5
        iterations = 4
        device = torch.device('cpu')
        lr = 0.00001
        experiment_name = "test"
        dataset_name = "synthetic_data"
        # data, _ = sklearn.datasets.make_blobs(n_samples=1000, n_features=2, centers=[(-25, -25), (-25, 25), (0,0), (25, -25), (25, 25)], random_state=1, cluster_std=1)
        data, _ = sklearn.datasets.make_blobs(n_samples=N, n_features=d, centers=blobs, random_state=1, cluster_std=1)
        data = data.astype('float32')
        # print(np.shape(data))
        # print(data)

        for r in range(1,R+1):
            index, time_per_r, build_time, memory_usage = build_index(batch_size, epochs, iterations, r, k, nr_neighbours, device, data, dataset_name, b, lr, experiment_name)
            inverted_indexes_paths, model_paths = zip(*index)
            # print(inverted_indexes_paths[0])
            inverted_index = None
            with open(inverted_indexes_paths[0], 'rb') as f:
                inverted_index = pickle.load(f)
            
            reconstructed_index = np.zeros(shape=(N), dtype=int)

            for i in range(b):
                items = inverted_index[i]
                for item in items:
                    reconstructed_index[item] = i

            plt.figure(figsize = (8,5))
            #TODO: try to plot per item instead of per bucket, to avoid 'striping'
            for i in range(N):
                bucket = reconstructed_index[i]
                plt.scatter(data[i][0], data[i][1], c=list(mcolors.XKCD_COLORS.values())[bucket], s=1)
            
            if not os.path.exists("synthetic_data_results"):
                os.mkdir("synthetic_data_results")

            plt.savefig(f"synthetic_data_results/N{N}_d{d}_blobs{blobs}_lr{lr}_R{r}_b{b}_epochs{epochs}_itr{iterations}_nbrs{nr_neighbours}.png", format='png')
=======
        b = 32
        k = 2
        r = 1
        nr_neighbours = 2
        batch_size = 256
        epochs = 5
        iterations = 20
        device = torch.device('cpu')
        lr = 0.001
        experiment_name = "test"
        dataset_name = "synthetic_data"
        data, _ = sklearn.datasets.make_blobs(n_samples=1000, n_features=2, centers=5, random_state=1, cluster_std=0.5)
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
>>>>>>> 97fb4c0 (Add separate test file for synthetic data visualizations)


if __name__ == "__main__":
    unittest.main()