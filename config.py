import torch

class Config:
    '''
    Class to set configuration for an experiment. Most parameters have default values which can be overwritten.
    '''
    def __init__(self, dataset_name, r= 4, k= 2, m = 2, freq_threshold = 2, 
                 epochs = 5, iterations = 4, batch_size = 1024, nr_train_neighbours = 100, 
                 nr_ann = 10, b = 0, lr = 0.001, reass_mode = 0, reass_chunk_size = 5000, 
                 query_batch_size = 2048, query_batched=True, datasize=1, 
                 mem_tracking = False, query_twostep=False, query_twostep_limit=10000, rp_dim=8, rp_seed=42):
        self.dataset_name = dataset_name
        self.r = r
        self.k = k
        self.m = m
        self.b = b
        self.lr = lr
        self.reass_mode = reass_mode
        self.reass_chunk_size = reass_chunk_size
        self.query_twostep = query_twostep
        self.query_twostep_limit = query_twostep_limit
        self.rp_dim = rp_dim
        self.rp_seed = rp_seed
        self.query_batched = query_batched
        self.freq_threshold = freq_threshold
        self.epochs = epochs
        self.iterations = iterations
        self.batch_size = batch_size
        self.nr_train_neighbours = nr_train_neighbours
        self.nr_ann = nr_ann
        self.datasize = datasize
        self.device = get_best_device()
        self.experiment_name = None
        self.mem_tracking = mem_tracking
        self.query_batch_size = query_batch_size

def get_best_device():
    '''
    Get the best available torch device (gpu if available, otherwise cpu).
    '''
    if torch.cuda.is_available():
        # covers both NVIDIA CUDA and AMD ROCm
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        # covers apple silicon mps
        return torch.device("mps") 
    else:
        return torch.device("cpu")
