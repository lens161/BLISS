from utils import get_best_device

class Config:
    def __init__(self, dataset_name, r= 4, k= 2, m = 2, freq_threshold = 2, epochs = 5, iterations = 20, batch_size = 256, nr_neighbours = 100, b = 0, lr = 0.001):
        self.dataset_name = dataset_name
        self.R = r
        self.K = k
        self.M = m
        self.B = b
        self.LR = lr
        self.FREQ_THRESHOLD = freq_threshold
        self.EPOCHS = epochs
        self.ITERATIONS = iterations
        self.BATCH_SIZE = batch_size
        self.NR_NEIGHBOURS = nr_neighbours
        self.device = get_best_device()