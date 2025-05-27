import numpy as np
import os
import random
import sklearn.datasets
import unittest
from sklearn.model_selection import train_test_split as sklearn_train_test_split

import bliss
import datasets as ds
import query
import train
import utils as ut
from bliss_model import BLISS_NN, BLISSDataset
from config import Config, get_best_device

def generate_random_array(size: int, dimensions: int, centers: int):
    '''
    generate random training and test arrays for unit tests
    '''
    X, _ = sklearn.datasets.make_blobs(n_samples=size, n_features=dimensions, centers=centers, random_state=1)
    return X

def generate_random_array_with_train_test_split(size: int, dimensions: int, centers: int, test_size: int):
    '''
    generate random training and test arrays for unit tests
    '''
    X, _ = sklearn.datasets.make_blobs(n_samples=size, n_features=dimensions, centers=centers, random_state=1)
    train, test = train_test_split(X, test_size=test_size)
    return train, test

def train_test_split(X, test_size):
    '''
    split an nd.array into training and test data
    '''
    # dimension = dimension if not None else X.shape[1]
    print(f"Splitting into train/test")
    return sklearn_train_test_split(X, test_size=test_size, random_state=1)

class TestTrainMethods(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.mnist_dataset_obj = ut.get_dataset_obj("mnist-784-euclidean", 1)
        cls.mnist_dataset_obj.prepare()
        cls.mnist_train = cls.mnist_dataset_obj.get_dataset()
        cls.mnist_test = cls.mnist_dataset_obj.get_queries()
        cls.mnist_nbrs, _ = cls.mnist_dataset_obj.get_groundtruth()

    def test_get_B(self):
        '''
        Test whether B value is correct according to our specification.
        '''
        self.assertEqual(ut.get_B(1000000), 1024) # SIFT
        self.assertEqual(ut.get_B(60000), 256) # MNIST
        self.assertEqual(ut.get_B(1000000000), 32768) # Billion-scale
        self.assertEqual(ut.get_B(1), 1) # Single item
        with self.assertRaises(Exception): # Empty dataset
            ut.get_B(0)

    def test_read_dataset(self):
        '''
        Test whether reading in the dataset gives us an np array of the correct shape.
        '''
        self.assertEqual(self.mnist_train.shape[0], 60000) # nr train items
        self.assertEqual(self.mnist_test.shape[0], 10000) # nr test items
        self.assertEqual(self.mnist_train.shape[1], 784) # dimensions
        self.assertEqual(self.mnist_test.shape[1], 784) # dimensions
        self.assertEqual(self.mnist_nbrs.shape[0], 10000) # nr test items
        self.assertEqual(self.mnist_nbrs.shape[1], 100) # nbrs per test item

    def test_get_nearest_neighbours_small(self):
        '''
        Test whether we find the correct nearest neighbours in a sample array.
        '''
        data = np.array([[1], [5], [88], [100], [125], [130], [132], [150], [273], [500]])
        nbrs = ut.get_nearest_neighbours_within_dataset(data, 2)
        self.assertTrue(np.array_equal(nbrs[0], [1, 2]))
        self.assertTrue(np.array_equal(nbrs[1], [0, 2]))
        self.assertTrue(np.array_equal(nbrs[2], [3, 4]))
        self.assertTrue(np.array_equal(nbrs[3], [2, 4]))
        self.assertTrue(np.array_equal(nbrs[4], [5, 6]))
        self.assertTrue(np.array_equal(nbrs[5], [6, 4]))
        self.assertTrue(np.array_equal(nbrs[6], [5, 4]))
        self.assertTrue(np.array_equal(nbrs[7], [6, 5]))
        self.assertTrue(np.array_equal(nbrs[8], [7, 6]))
        self.assertTrue(np.array_equal(nbrs[9], [8, 7]))
   
    def test_get_nearest_neighbours(self):
        '''
        Test whether our nearest neighbour function finds the same nearest neighbours for test data in a real dataset.
        '''
        nbrs_test = ut.get_nearest_neighbours_in_different_dataset(self.mnist_train, self.mnist_test, 100)
        self.assertTrue(np.array_equal(np.sort(nbrs_test), np.sort(self.mnist_nbrs)))
        
    def test_get_nearest_neighbours_from_file(self):
        '''
        Test whether fetching neighbours from a file gives the same array as when computing them.
        '''
        dataset_name = "mnist-784-euclidean"
        amount = 100
        sample_size = 60000
        # check that the file exists already, otherwise the test will just compute neighbours twice
        self.assertTrue(os.path.exists(f"data/{dataset_name}-nbrs{amount}-sample{sample_size}.csv"))
        file_nbrs = ut.get_train_nearest_neighbours_from_file(self.mnist_train, amount, sample_size, dataset_name)
        fresh_nbrs = ut.get_nearest_neighbours_within_dataset(self.mnist_train, amount)
        self.assertTrue(np.array_equal(np.sort(file_nbrs), np.sort(fresh_nbrs)))

    def test_make_ground_truth_labels_small(self):
        '''
        Test whether ground truth labels are computed correctly.
        '''
        B = 3
        data = np.array([[1], [5], [88], [100], [125], [130], [132], [150], [273], [500]])
        index = np.array([0, 0, 2, 1, 0, 1, 1, 2, 2, 0])
        nbrs = ut.get_nearest_neighbours_within_dataset(data, 2)
        device = get_best_device()
        labels = ut.make_ground_truth_labels(B, nbrs, index, 10, device)
        self.assertTrue(np.array_equal(labels[0], [1, 0, 1]))
        self.assertTrue(np.array_equal(labels[1], [1, 0, 1]))
        self.assertTrue(np.array_equal(labels[2], [1, 1, 0]))
        self.assertTrue(np.array_equal(labels[3], [1, 0, 1]))
        self.assertTrue(np.array_equal(labels[4], [0, 1, 0]))
        self.assertTrue(np.array_equal(labels[5], [1, 1, 0]))
        self.assertTrue(np.array_equal(labels[6], [1, 1, 0]))
        self.assertTrue(np.array_equal(labels[7], [0, 1, 0]))
        self.assertTrue(np.array_equal(labels[8], [0, 1, 1]))
        self.assertTrue(np.array_equal(labels[9], [0, 0, 1]))

    def test_initial_bucket_assignment_small(self):
        '''
        Test whether initial assignment arrays are of the correct shape and that the assignment is somewhat fairly distributed.
        '''
        N = 10000
        r = 1
        B = ut.get_B(N)
        index, counts = bliss.assign_initial_buckets(N, r, B)
        self.assertEqual(len(index), N)
        self.assertEqual(len(counts), B)

        expected_bucketsize = N/B
        min_bucketsize = expected_bucketsize/2
        max_bucketsize = expected_bucketsize*2
        self.assertFalse((counts > max_bucketsize).any())
        self.assertFalse((counts < min_bucketsize).any())
    
    def test_invert_index_small(self):
        testindex = np.array([0, 2, 0, 1, 1, 2, 0, 1, 2, 1])
        test_buckets = np.unique(testindex, return_counts=True)[1]
        test_size = testindex.shape[0]
        inverted_index, offsets = bliss.invert_index(testindex, test_buckets, test_size)
        self.assertEqual(offsets[0], test_buckets[0])
        self.assertEqual(offsets[1]-offsets[0], test_buckets[1])
        self.assertEqual(offsets[2]-offsets[1], test_buckets[2])
        self.assertTrue(np.array_equal(inverted_index[0:offsets[0]], np.array([0, 2, 6])))
        self.assertTrue(np.array_equal(inverted_index[offsets[0]:offsets[1]], np.array([3, 4, 7, 9])))
        self.assertTrue(np.array_equal(inverted_index[offsets[1]:offsets[2]], np.array([1, 5, 8])))

    #TODO: replace with getting training sample from memmap

    # def test_get_sample(self):
    #     data_small = np.array([[1], [5], [88], [100], [125], [130], [132], [150], [273], [500]])
    #     SIZE_small = 10
    #     DIMENSION_small = 1

    #     sample_small, rest_small, sample_small_size, rest_small_size, train_bool_small = get_sample(data_small, SIZE_small, DIMENSION_small)
    #     self.assertEqual(np.shape(sample_small)[0], SIZE_small)
    #     self.assertEqual(sample_small_size, SIZE_small)
    #     self.assertEqual(np.shape(rest_small), ())
    #     self.assertEqual(rest_small_size, 0)
    #     self.assertEqual(train_bool_small, True)

    #     data_large = np.zeros((11_000_000, 1))
    #     SIZE_large = 11_000_000
    #     DIMENSION_large = 1
    #     sample_large, rest_large, sample_large_size, rest_large_size, train_bool_large = get_sample(data_large, SIZE_large, DIMENSION_large)
    #     self.assertEqual(np.shape(sample_large)[0], SIZE_large*0.01)
    #     self.assertEqual(sample_large_size, SIZE_large*0.01)
    #     self.assertEqual(np.shape(rest_large)[0], SIZE_large*0.99)
    #     self.assertEqual(rest_large_size, SIZE_large*0.99)
    #     self.assertEqual(train_bool_large, False)

    
if __name__ == "__main__":
    unittest.main()