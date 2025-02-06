import numpy as np
import sklearn.datasets
import h5py

def generate_random_array(size: int, dimensions: int, centers: int):
    X, _ = sklearn.datasets.make_blobs(n_samples=size, n_features=dimensions, centers=centers, random_state=1)
    #X_train, X_test = train_test_split(X, test_size=0.1)
    #write_output(X_train, X_test, out_fn, distance)