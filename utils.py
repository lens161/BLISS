import logging
import math
import matplotlib.pyplot as plt # type: ignore
import numpy as np
import os
import psutil
import torch
from torch.amp import autocast
from faiss import IndexFlatL2, IndexPQ, IndexIVFPQ, vector_to_array

import datasets as ds
from bliss_model import BLISS_NN


######################################################################
# Helpers for getting ground truth for train vectors or query vectors.
######################################################################

def get_nearest_neighbours_within_dataset(dataset, amount):
    '''
    Find the true nearest neighbours of vectors within a dataset. To avoid returning a datapoint as its own neighbour, we search for amount+1 neighbours and then filter out
    the first vector (ordered by distance so the vector itself should be the first point).
    '''
    nbrs = np.zeros((len(dataset), amount), dtype=np.int32)
    nbrs_index = IndexFlatL2(dataset.shape[1])
    nbrs_index.add(dataset)
    chunk_size = 100_000
    chunks = math.ceil(len(dataset) / chunk_size)
    start = 0
    for i in range(0, chunks):
        end = min(len(dataset), start+chunk_size)
        _, I = nbrs_index.search(dataset[start:end], amount+1)
        I = I[:, 1:]
        I = I.astype(np.int32)
        nbrs[start:end] = I
        start = end
    return nbrs

def get_nearest_neighbours_in_different_dataset(dataset, queries, amount):
    '''
    Find the true nearest neighbours of query vectors in dataset. No filtering needed here because the queries do not appear in the dataset.
    '''
    nbrs = IndexFlatL2(dataset.shape[1])
    nbrs.add(dataset)
    _, I = nbrs.search(queries, amount)
    I = np.asarray(I, dtype=np.int32)
    return I

def get_train_nearest_neighbours_from_file(dataset, amount, sample_size, dataset_name, datasize):
    '''
    Helper to read/write nearest neighbour of train data to file so we can test index building without repeating preprocessing each time.
    Should not be used in actual algorithm or experiments where timing the preprocessing is important.
    '''
    if not os.path.exists(f"data/{dataset_name}-size{datasize}-nbrs{amount}-sample{sample_size}.npy"):
        filename = f"data/{dataset_name}-size{datasize}-nbrs{amount}-sample{sample_size}.npy"
        print(f"no nbrs file found for {dataset_name} with amount={amount} and samplesize={sample_size}, calculating {amount} nearest neighbours")
        logging.info("No neighbours file found, calculating ground truths of training sample")
        I = get_nearest_neighbours_within_dataset(dataset, amount)
        print("writing neighbours to nbrs file")
        np.save(filename, I)

    else:
        print(f"found nbrs file for {dataset_name} with amount={amount} and samplesize={sample_size}, reading true nearest neighbours from file")
        logging.info("Reusing ground truths for training sample from file")
        filename = f"data/{dataset_name}-size{datasize}-nbrs{amount}-sample{sample_size}.npy"
        I = np.load(filename)
    return I

######################################################################
# Helpers for training index
######################################################################

def normalise_data(data):
    '''
    Normalize a dataset (divide vectors by their magnitude).
    '''
    norms = np.linalg.norm(data, axis=1, keepdims=True)
    data = data / norms
    return data

def get_training_sample(dataset: ds.Dataset, sample_size, SIZE, DIM):
    '''
    Get a training sample of sample_size, given a dataset. The sample is taken by selecting random indices.
    If the dataset is too large to load at once, it is loaded in chunks, and random indices are selected
    per chunk.
    '''
    if sample_size == SIZE:
        return dataset.get_dataset(), np.arange(0, SIZE)
    sample = np.zeros((sample_size, DIM))
    sample_indices = np.zeros(sample_size)
    chunk_size = 1_000_000
    chunk_sample_size = sample_size // chunk_size
    index = 0
    for i, batch in enumerate(dataset.get_dataset_iterator(bs=chunk_size)):
        random_order = np.arange(len(batch))
        np.random.seed(i)
        np.random.shuffle(random_order)
        chunk_sample_indices = np.sort(random_order[:chunk_sample_size])
        sample_indices[index : index+len(batch)] = chunk_sample_indices
        sample[index : index+len(batch)] = batch[chunk_sample_indices]
        index += len(batch)
    return sample, sample_indices

def get_training_sample_from_memmap(dataset: ds.Dataset, sample_size, SIZE, DIM, dataset_name, datasize):
    '''
    Given a dataset (as a memmap), sample data for model training.
    For small datasets, the full dataset is used to train.
    For large datasets, a random sample is taken across the dataset. It is assumed the sample size is small enough to load the sample into memory.
    '''
    sample = np.zeros(shape=(sample_size, DIM), dtype=np.float32)
    sample_filename = f"data/{dataset_name}_size{datasize}_sample{sample_size}.npy"
    if os.path.exists(sample_filename):
        sample = np.load(sample_filename)
    else:
        if sample_size!=SIZE:
            dataset_mmp = dataset.get_dataset_memmap()
            random_order = np.arange(SIZE)
            np.random.seed(42)
            np.random.shuffle(random_order)
            sample_indexes = np.sort(random_order[:sample_size])
            sample[:] = dataset_mmp[sample_indexes, :].copy()
            np.save(sample_filename, sample)
        else:
            data = dataset.get_dataset()
            if dataset.distance() == "angular":
                data = normalise_data(data)
            sample[:] = data
    return torch.from_numpy(sample)

def make_ground_truth_labels(B, neighbours, index, sample_size):
    '''
    DEPRECATED!
    Create ground truth labels for training sample, based on the set of nearest neighbours of each training vector. 
    A label is a B-dimensional vector, where each digit is either 0 (false) if that bucket does not contain any
    nearest neighbours of a vector, and 1 (true) if the bucket contains at least one nearest neighbour of that vector.
    '''
    # start = time.time()
    labels = np.zeros((sample_size, B), dtype=bool)
    # for each vector i create an array of amount of neighbours
    vectors = np.concatenate([np.full(len(n), i) for i, n in enumerate(neighbours)])
    # build column indices by applying the mapping to each neighbour array.
    buckets = np.concatenate([index[n] for n in neighbours])
    # set bucket entries to True.
    labels[vectors, buckets] = True
    return torch.from_numpy(labels).float()

def get_labels(neighbours, lookup, b, device=torch.device("cuda")):
    '''
    Create new labels tensor and set positions of per vector true buckets to one  
    '''
    batch_size = neighbours.shape[0]
    labels = torch.zeros((batch_size, b), dtype=torch.float32, device=device)
    # get buckets at neighbour indexes from lookup
    bucket_ids = lookup[neighbours]
    # add ones in label matrix at indices of buckets that have neighbours
    labels.scatter_(
        dim=1,
        index=bucket_ids,
        src=torch.ones_like(bucket_ids, dtype=torch.float32, device=device)
    )
    return labels 

def reassign_vector_to_bucket(index, bucket_sizes, candidates, item_index):
    '''
    Reassign a vector to the least occupied of the top-k buckets predicted by the model.
    '''
    candidate_sizes = bucket_sizes[candidates]
    best_bucket = candidates[np.argmin(candidate_sizes)]
    index[item_index] = best_bucket
    bucket_sizes[best_bucket] += 1

def assign_to_buckets_vectorised(bucket_sizes, SIZE, index, chunk_size, i, topk_per_vector, memory_tracking=False):
    '''
    Reassign a chunk of vectors to a new bucket. The vectors are reassigned to the least
    occupied of the top-k buckets predicted by the model, but as multiple vectors are reassigned at once,
    there is no guarantee that the selected bucket is the least occupied if multiple vectors
    are getting reassigned to the same bucket in a single batch.
    '''
    memory_usage = 0
    candidate_sizes_per_vector = bucket_sizes[topk_per_vector]    
    vectors = np.arange(topk_per_vector.shape[0])
    sizes = np.argmin(candidate_sizes_per_vector, axis=1)
    # get the least ocupied of each candidate set 
    least_occupied = topk_per_vector[vectors, sizes]
    if memory_tracking:
        process = psutil.Process(os.getpid())
        memory_usage = process.memory_full_info().uss / (1024 ** 2)
    index[i : min(i + chunk_size, SIZE)] = least_occupied

    bucket_increments = np.bincount(least_occupied, minlength=len(bucket_sizes))
    bucket_sizes[:] = np.add(bucket_sizes, bucket_increments)
    return memory_usage

# copy for rm 3 just to be shure using length of least occupied does not fuck up the other modes
def assign_to_buckets_vectorised_rm3(bucket_sizes, SIZE, index, chunk_size, i, topk_per_vector, memory_tracking=False):
    '''
    Reassign a chunk of vectors to a new bucket. The vectors are reassigned to the least
    occupied of the top-k buckets predicted by the model, but as multiple vectors are reassigned at once,
    there is no guarantee that the selected bucket is the least occupied if multiple vectors
    are getting reassigned to the same bucket in a single batch.
    '''
    memory_usage = 0
    candidate_sizes_per_vector = bucket_sizes[topk_per_vector]    
    vectors = np.arange(topk_per_vector.shape[0])
    sizes = np.argmin(candidate_sizes_per_vector, axis=1)
    # get the least ocupied of each candidate set 
    least_occupied = topk_per_vector[vectors, sizes]
    end = len(least_occupied)
    if memory_tracking:
        process = psutil.Process(os.getpid())
        memory_usage = process.memory_full_info().uss / (1024 ** 2)
    index[i : i+end] = least_occupied

    bucket_increments = np.bincount(least_occupied, minlength=len(bucket_sizes))
    bucket_sizes[:] = np.add(bucket_sizes, bucket_increments)
    return memory_usage

def get_all_topk_buckets(loader, k, candidate_buckets, map_model, offset, device,  mem_tracking = False):
    '''
    Prepare a table with the top-k buckets for all vectors in a loader according to the model.
    '''
    logging.info(f"Mapping all train vectors to buckets")
    start_idx = offset
    memory = 0
    with torch.no_grad():
        for batch_data, _, in loader:
            batch_size = len(batch_data)
            batch_candidate_buckets, current_mem = get_topk_buckets_for_batch(batch_data, k, map_model, device, mem_tracking)
            if mem_tracking:
                memory = current_mem if current_mem>memory else memory
            batch_candidate_buckets.numpy()
            candidate_buckets[start_idx : start_idx + batch_size, :] = batch_candidate_buckets
            start_idx += batch_size
    return memory

def get_topk_buckets_for_batch(batch_data, k, map_model, device, mem_tracking=False):
    '''
    Prepare a table with the top-k buckets for a batch of vectors according to the model.
    '''
    if mem_tracking:
        process = psutil.Process(os.getpid())
    memory=0
    batch_data = batch_data.to(device)
    # only do autocast (mixed precision) on cuda devices
    if device == torch.device("cuda"):
        with torch.no_grad(), autocast("cuda"):
            logits = map_model(batch_data)
            bucket_probabilities = torch.sigmoid(logits)
    else:
        with torch.no_grad():
            logits = map_model(batch_data)
            bucket_probabilities = torch.sigmoid(logits)

    bucket_probabilities_cpu = bucket_probabilities.cpu()
    _, candidate_buckets = torch.topk(bucket_probabilities_cpu, k, dim=1)

    if mem_tracking:
        if device == torch.device("cuda") or device == torch.device("mps"):
            memory = torch.cuda.memory_allocated(device) / (1024**2)
        else:
            memory = process.memory_full_info().uss / (1024 ** 2)
    del bucket_probabilities, bucket_probabilities_cpu
    if device == torch.device("cuda"):
        torch.cuda.empty_cache()
    return candidate_buckets, memory

def get_dataset_obj(dataset_name, size):
    '''
    Return a dataset object 
    '''
    if dataset_name == "bigann":
        return ds.BigANNDataset(size)
    elif dataset_name == "Deep1B":
        return ds.Deep1BDataset(size)
    elif dataset_name == "Yandex":
        return ds.Text2Image1B(size)
    elif dataset_name == "MSSpaceV":
        return ds.MSSPACEV1B(size)
    elif dataset_name == "sift-128-euclidean":
        return ds.Sift_128()
    elif dataset_name == "glove-100-angular":
        return ds.Glove_100()
    elif dataset_name == "mnist-784-euclidean":
        return ds.Mnist_784()
    else:
        print("dataset not supported yet")

def get_B(n):
    '''
    Calculated suggested B (nr of buckets) based on the size of the dataset. Recommended B is the first power of 2 larger than sqrt(n).
    '''
    if n > 0:
        sq = math.sqrt(n)
        B = 2 ** round(math.log(sq, 2))
        return B
    else:
        raise Exception(f"cannot calculate B for empty dataset!")

######################################################################
# Helpers for loading and saving models and indexes.
######################################################################

def save_model(model, dataset_name, r, R, K, B, lr, batch_size, reass_mode, chunk_size, e, i):
    '''
    Save a (trained) model in the models folder and return the path.
    '''
    model_name = f"model_{dataset_name}_r{r}_k{K}_b{B}_lr{lr}"
    directory = f"models/{dataset_name}_r{R}_k{K}_b{B}_lr{lr}_bs={batch_size}_reass={reass_mode}_chunk_size={chunk_size}_e={e}_i={i}/"
    MODEL_PATH = os.path.join(directory, f"{model_name}.pt")
    os.makedirs(directory, exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)
    file_size = os.path.getsize(MODEL_PATH) / 1024**2
    return MODEL_PATH, file_size

def load_model(model_path, dim, b):
    '''
    Load a (trained) model from the specified path for inference (weights only).
    '''
    inf_device = torch.device("cpu")
    model = BLISS_NN(dim, b)
    model.load_state_dict(torch.load(model_path, weights_only=True, map_location=inf_device))
    model.eval()
    return model

def save_inverted_index(inverted_index, offsets, dataset_name, model_num, R, K, B, lr, batch_size, reass_mode, chunk_size, e, i):
    '''
    Save an inverted index (for a specific dataset and parameter setting combination) in the models folder and return the path.
    '''
    index_name = f"index_model{model_num}_{dataset_name}_r{model_num}_k{K}_b{B}_lr{lr}"
    offsets_name = f"offsets_model{model_num}_{dataset_name}_r{model_num}_k{K}_b{B}_lr{lr}"
    directory = f"models/{dataset_name}_r{R}_k{K}_b{B}_lr{lr}_bs={batch_size}_reass={reass_mode}_chunk_size={chunk_size}_e={e}_i={i}/"
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
    index_path = os.path.join(directory, f"{index_name}.npy")
    offsets_path = os.path.join(directory, f"{offsets_name}.npy")
    np.save(index_path, inverted_index)
    np.save(offsets_path, offsets)
    index_size = (os.path.getsize(index_path) + os.path.getsize(offsets_path)) / 1024**2
    return index_path, index_size

######################################################################
# Helpers for plots created during index building and collecting statistics.
######################################################################

def make_loss_plot(learning_rate, iterations, epochs_per_iteration, k, B, experiment_name, all_losses, shuffle, reass_mode):
    '''
    Plot the total loss of the model after each epoch.
    '''
    foldername = f"results/{experiment_name}"
    if not os.path.exists("results"):
        os.mkdir("results")
    if not os.path.exists(foldername):
        os.mkdir(foldername)
    plt.figure(figsize=(10, 5))
    plt.plot(all_losses, marker='.')
    plt.title('Training Loss Over Epochs')
    plt.xlabel('Epoch (accumulated over iterations)')
    plt.ylabel('Average Loss')
    plt.grid(True)
    plt.savefig(f"{foldername}/training_loss_lr={learning_rate}_I={iterations}_E={epochs_per_iteration}_k{k}_B{B}_shf={shuffle}_reass={reass_mode}.png")

def calc_load_balance(bucket_size_stats):
    '''
    Calculate the mean load balance of the bucket assignments of all r indexes for a particular dataset.
    Load balance of a single index is defined as the inverse of the standard deviation of the bucket sizes of that particular index (1-standard deviation).
    Combined load balance for a dataset is then the mean across load balance of r indexes.
    '''
    load_balance_per_model = []
    for r in bucket_size_stats:
        load_balance = 1 / np.std(r)
        load_balance_per_model.append(load_balance)

    avg_load_balance = np.mean(load_balance_per_model)
    return avg_load_balance

def recall(results, neighbours):
    '''
    Calculate mean recall for a set of queries.
    '''
    recalls = np.zeros(len(results), dtype=np.float32)
    for i, (ann, nn) in enumerate(zip(results, neighbours)):
        recalls[i] = recall_single(ann, nn)
    return np.mean(recalls)

def recall_single(results, neighbours):
    '''
    Calculate recall for an individual query.
    '''
    return len(set(results) & set(neighbours))/len(neighbours)


######################################################################
# Other helper functions.
######################################################################

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
    
def set_torch_seed(seed, device):
    '''
    Set torch seed for ease of reproducibility during testing.
    '''
    torch.manual_seed(seed)
    if device == torch.device("cuda"):
        torch.cuda.manual_seed(seed)
    elif device == torch.device("mps"):
        torch.mps.manual_seed(seed)

def train_ivfpq(training_data: np.ndarray, data = None, m = 8, nbits = 8, nlist=256):
    '''
    Train a PQ index. Training date is encoded into the provided number of bits.
    If any additional data is passed, it is added to the pq index instead of only adding the train data.
    '''
    d = training_data.shape[1]
    quantiser = IndexFlatL2(d)
    ivf_pq = IndexIVFPQ(quantiser, d, nlist, m, nbits)
    ivf_pq.train(training_data)
    if data is not None:
        ivf_pq.add(data)
    else:
        ivf_pq.add(training_data)
    return (ivf_pq, m)

def random_projection(X, target_dim):
    '''
    Make a random projection of a set of input vectors, by multiplying with a random vector of the target dimension.
    '''
    original_dim = X.shape[1]
    R = np.random.randn(original_dim, target_dim) / np.sqrt(target_dim)
    return np.dot(X, R)

def norm_ent(bucket_sizes):
    '''
    Get normalised entropy to check balancedness of buckets.
    '''
    B = len(bucket_sizes)
    total = sum(bucket_sizes)
    probs = np.zeros(B, dtype=np.float32)
    probs = bucket_sizes/total
    shann_entropy = - sum(probs[probs>0]*np.log(probs[probs>0]))
    norm_entropy = shann_entropy/math.log(B)
    return norm_entropy