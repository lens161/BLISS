from utils import get_best_device

class Config:
    def __init__(self, dataset_name, r= 4, k= 2, m = 2, freq_threshold = 2, epochs = 5, iterations = 4, batch_size = 256, nr_neighbours = 100, b = 0, lr = 0.001, global_reass=False, shuffle = False, datasize=1):
        self.dataset_name = dataset_name
        self.r = r
        self.k = k
        self.m = m
        self.b = b
        self.lr = lr
        self.global_reass = global_reass
        self.shuffle = shuffle
        self.freq_threshold = freq_threshold
        self.epochs = epochs
        self.iterations = iterations
        self.batch_size = batch_size
        self.nr_neighbours = nr_neighbours
        self.datasize = datasize
        self.device = get_best_device()
        self.experiment_name = None
        self.memlog_path = None