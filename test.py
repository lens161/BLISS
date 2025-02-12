import numpy as np
import sklearn.datasets
import unittest
from sklearn.model_selection import train_test_split as sklearn_train_test_split
from utils import *
from bliss import *

def generate_random_array(size: int, dimensions: int, centers: int):
    '''
    generate random training and test arrays for unit tests
    '''
    X, _ = sklearn.datasets.make_blobs(n_samples=size, n_features=dimensions, centers=centers, random_state=1)
    train, test = train_test_split(X, test_size=0.1)
    return train, test

def train_test_split(X, test_size):
    '''
    split an nd.array into training and test data
    '''
    # dimension = dimension if not None else X.shape[1]
    print(f"Splitting into train/test")
    return sklearn_train_test_split(X, test_size=test_size, random_state=1)

class TestTrainMethods(unittest.TestCase):

    mnist_train, mnist_test, mnist_nbrs = read_dataset("mnist-784-euclidean")

    def test_get_B(self):
        self.assertEqual(get_B(1000000), 1024) # SIFT
        self.assertEqual(get_B(60000), 256) # MNIST
        self.assertEqual(get_B(1000000000), 32768) # Billion-scale
        self.assertEqual(get_B(1), 1) # Single item
        with self.assertRaises(Exception): # Empty dataset
            get_B(0)

    def test_read_dataset(self):
        self.assertEqual(self.__class__.mnist_train.shape[0], 60000) # nr train items
        self.assertEqual(self.__class__.mnist_test.shape[0], 10000) # nr test items
        self.assertEqual(self.__class__.mnist_train.shape[1], 784) # dimensions
        self.assertEqual(self.__class__.mnist_test.shape[1], 784) # dimensions
        self.assertEqual(self.__class__.mnist_nbrs.shape[0], 10000) # nr test items
        self.assertEqual(self.__class__.mnist_nbrs.shape[1], 100) # nbrs per test item
    
    def test_get_nearest_neighbours(self):
        nbrs_test = get_nearest_neighbours_faiss_in_different_dataset(self.__class__.mnist_train, self.__class__.mnist_test, 100)
        self.assertTrue(np.array_equal(np.sort(nbrs_test), np.sort(self.__class__.mnist_nbrs)))
    
    def test_get_nearest_neighbours_from_file(self):
        dataset_name = "mnist-784-euclidean"
        amount = 100
        # check that the file exists already, otherwise the test will just compute neighbours twice
        self.assertTrue(os.path.exists(f"data/{dataset_name}-trainnbrs-{amount}.csv"))
        file_nbrs = get_train_nearest_neighbours_from_file(self.__class__.mnist_train, amount, dataset_name)
        fresh_nbrs = get_nearest_neighbours_faiss_within_dataset(self.__class__.mnist_train, amount)
        self.assertTrue(np.array_equal(np.sort(file_nbrs), np.sort(fresh_nbrs)))
        
    
    
if __name__ == "__main__":
    unittest.main()