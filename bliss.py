import numpy as np
import sys
from config import Config
import time
import torch
import statistics
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.utils import murmurhash3_32 as mmh3
from utils import *
import psutil  # type: ignore
import os
from functools import partial
from multiprocessing import Pool
import logging
import faiss

class BLISSDataset(Dataset):
    def __init__(self, data, labels, device):
        self.device = device
        self.labels = labels
        if device == torch.device("cpu"):
            self.data = data
        else:
            self.data = torch.from_numpy(data).float()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # turn nd.array into tensor when fetched from the Dataset
        if self.device == torch.device("cpu"):
            vector = torch.from_numpy(self.data[idx]).float()
            label = torch.from_numpy(self.labels[idx]).float()
        else:
            vector = self.data[idx]
            label = self.labels[idx]
        return vector, label, idx

class BLISS_NN(nn.Module):
    def __init__(self, input_size, output_size):
        super(BLISS_NN, self).__init__()
        # takes input and projects it to 512 hidden neurons
        # fc stands for fully connected, referring to a fully connected matrix being created
        self.fc1 = nn.Linear(input_size, 512)
        # activation function
        self.relu = nn.ReLU()
        # output layer maps 512 hidden neurons to output neurons (representing the buckets)
        self.fc2 = nn.Linear(512, output_size)
        # turns all output values into softmax values that sum to 1 -> probabilities
        # self.sigmoid = nn.Sigmoid(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)

        return x

def train_model(model, dataset, index, sample_size, bucket_sizes, neighbours, r, config: Config):
    model.to(config.device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    g = torch.Generator()
    g.manual_seed(r)
    train_loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, num_workers=8, generator=g)
    all_losses = []
    for i in range(config.iterations):
        model.train() 
        for epoch in range(config.epochs):
            epoch_losses = []
            print(f"training epoch ({i}, {epoch})")
            logging.info(f"Training epoch ({i}, {epoch})")
            start = time.time()
            for batch_data, batch_labels, _ in train_loader:
                batch_data = batch_data.to(config.device)
                batch_labels = batch_labels.to(config.device)
                optimizer.zero_grad()
                probabilities = model(batch_data)
                loss = criterion(probabilities, batch_labels)
                loss.backward()
                optimizer.step()
                epoch_losses.append(loss.item())
            finish = time.time()
            elapsed = finish-start
            print(f"epoch {epoch} took {elapsed}")
            avg_loss = statistics.mean(epoch_losses)
            print(f"epoch {epoch} avg. loss = {avg_loss}", flush=True)
            all_losses.append(avg_loss)
        if config.global_reass:
            global_reassign_buckets(model, dataset, index, neighbours, bucket_sizes, config)
        else:
            reassign_buckets(model, dataset, index, bucket_sizes, sample_size, neighbours, config)
        np.set_printoptions(threshold=6, suppress=True)
        print(f"index after iteration {i}: \r{index}", flush=True)

    make_loss_plot(config.lr, config.iterations, config.epochs, config.k, config.b, config.experiment_name, all_losses, config.shuffle, config.global_reass)
 
def reassign_buckets(model, dataset, index, bucket_sizes, sample_size, neighbours, config: Config):
    logging.info(f"Reassigning vectors to new buckets")
    sample_size, _ = np.shape(dataset.data)
    model.to("cpu")
    model.eval()
    reassign_loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=config.shuffle, num_workers=8)
    bucket_sizes[:] = 0

    start = time.time()
    with torch.no_grad():
        for batch_data, batch_labels, batch_indices in reassign_loader:
            batch_data = batch_data.to("cpu")
            bucket_probabilities = torch.sigmoid(model(batch_data))

            for probability_vector, idx in zip(bucket_probabilities, batch_indices):
                reassign_vector_to_bucket(probability_vector, index, bucket_sizes, config.k, idx)
                     
    finish = time.time()
    elapsed = finish - start
    process = psutil.Process(os.getpid())
    mem_usage = process.memory_info().rss / (1024 ** 2)
    log_mem(f"shuffle={config.shuffle}_reassign_buckets", mem_usage, config.memlog_path)

    print(f"Memory usage reassign batched: {mem_usage:.2f} MB")
    print(f"reassigning took {elapsed}", flush=True)
    new_labels = make_ground_truth_labels(config.b, neighbours, index, sample_size, config.device)
    dataset.labels = new_labels
    model.to(config.device)

def reassign_vector_to_bucket(probability_vector, index, bucket_sizes, k, item_index):
    value, indices_of_topk_buckets = torch.topk(probability_vector, k)
    # get sizes of candidate buckets
    candidate_sizes = bucket_sizes[indices_of_topk_buckets]
    # get bucket at index of smallest bucket from bucket_sizes
    best_bucket = indices_of_topk_buckets[np.argmin(candidate_sizes)]
    index[item_index] = best_bucket
    bucket_sizes[best_bucket] +=1  

def global_reassign_buckets(model, dataset, index, neighbours, bucket_sizes, config: Config):

    model.eval()
    data_loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False, num_workers=8)
    
    all_predictions = []

    start = time.time()
    with torch.no_grad():
        for batch_data, _, batch_indices in data_loader:
            batch_data = batch_data.to(config.device)
            probabilities = torch.sigmoid(model(batch_data))
            all_predictions.append(probabilities.cpu())
    
    # concatenate all predictions along the 0th dimension -> create tensor of predictions per vector of shape(N, B)
    all_predictions = torch.cat(all_predictions, dim=0) 
    process = psutil.Process(os.getpid())
    mem_usage = process.memory_info().rss / (1024 ** 2)
    print(f"global ress memory usage: {mem_usage:.2f} MB")
    log_mem("global_reassign_buckets", mem_usage, config.memlog_path)
    N = all_predictions.size(0)
    
    bucket_sizes[:] = 0
    
    for i in range(N):
        probs = all_predictions[i]
        _, candidate_buckets = torch.topk(probs, config.k)
        candidate_buckets = candidate_buckets.numpy()
        candidate_sizes = bucket_sizes[candidate_buckets]
        best_candidate = candidate_buckets[np.argmin(candidate_sizes)]
        index[i] = best_candidate
        bucket_sizes[best_candidate] += 1
    
    finish = time.time()
    elapsed = finish - start

    print(f"reassigning took {elapsed}")
    new_labels = make_ground_truth_labels(config.b, neighbours, index, N, config.device)
    dataset.labels = new_labels
    print("New bucket sizes:", bucket_sizes)


def reassign_buckets_vectorized(model, dataset, index, bucket_sizes, sample_size, neighbours,device, config: Config):
    sample_size, _ = np.shape(dataset.data)
    model.to(config.device)
    model.eval()
    reassign_loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False, num_workers=8)
    
    # create auxiliary tensor representing bucket_sizes
    bucket_sizes_t = torch.zeros(config.B, device=device, dtype=torch.int32)
    
    global_item_index = 0 # index from list "index" on cpu
    start = time.time()
    with torch.no_grad():
        for batch_data, _ in reassign_loader:
            batch_data = batch_data.to(config.device)
            bucket_probabilities = torch.sigmoid(model(batch_data))
            _, candidate_indices = torch.topk(bucket_probabilities, config.k, dim=1)
            # get current sizes for candidate buckets (broadcast bucket_sizes_t)
            candidate_sizes = bucket_sizes_t[candidate_indices]  # shape: (batch_size, k)
            #for each vector, choose the candidate with the smallest bucket count
            chosen_candidate_idx = torch.argmin(candidate_sizes, dim=1)
            #get the chosen buckets for batch
            chosen_buckets = candidate_indices[torch.arange(candidate_indices.size(0)), chosen_candidate_idx]
            
            # batched update bucket sizes
            ones = torch.ones_like(chosen_buckets, dtype=torch.int32, device=config.device)
            bucket_sizes_t = bucket_sizes_t.scatter_add(0, chosen_buckets, ones)
            
            # send chosen buckets to cpu
            chosen_buckets_cpu = chosen_buckets.cpu().numpy()
            for bucket in chosen_buckets_cpu:
                index[global_item_index] = bucket
                global_item_index += 1
    end = time.time()
    
    #update global bucket_sizes list
    bucket_sizes[:] = bucket_sizes_t.cpu().numpy()
    
    process = psutil.Process(os.getpid())
    mem_usage = process.memory_info().rss
    print(f"Memory usage: {mem_usage / (1024 ** 2):.2f} MB")
    print(f"reassigning vect. took: {end - start:.4f} seconds")
    
    new_labels = make_ground_truth_labels(config.b, neighbours, index, sample_size, config.device)
    dataset.labels = new_labels
            
def assign_initial_buckets(train_size, rest_size, r, B):
    '''
    assign bucket labels to vectors (indeces in the nd.array) using a hash function.
    the hash fucntion used here is the same as in the original code from the BLISS github.
    TO-DO: add reference link
    '''
    index = np.zeros(train_size, dtype=np.uint32) # from 0 to train_size-1
    bucket_sizes = np.zeros(B, dtype=np.uint32)

    for i in range(train_size):
        bucket = mmh3(i,seed=r)%B
        index[i] = bucket
        bucket_sizes[bucket] += 1
    
    return np.array(index), bucket_sizes

def make_ground_truth_labels(B, neighbours, index, sample_size, device):

    labels = np.zeros((sample_size, B), dtype=bool)

    for i in range(sample_size):
        for neighbour in neighbours[i]:
            bucket = index[neighbour]
            labels[i, bucket] = True
    if device != torch.device("cpu"):
        labels = torch.from_numpy(labels).float()
    return labels

def map_all_to_buckets(rst_vectors, rest_indexes, k, bucket_sizes, index, model_path, training_sample_size, DIMENSION, B):
    rst_vectors = torch.from_numpy(rst_vectors).float()
    print(f"training sample size = {training_sample_size}")
    logging.info(f"Mapping all train vectors to buckets")
    map_model = BLISS_NN(DIMENSION, B)
    # print("loading model")
    map_model.load_state_dict(torch.load(model_path, weights_only=True))
    map_model.eval()
    # print("finished loading model")

    for i, vector in enumerate(rst_vectors):
        # if i < training_sample_size:
        #     print("wrong start")
        scores = map_model(vector)
        probabilities = torch.sigmoid(scores)
        values, candidates = torch.topk(probabilities, k)
        smallest_bucket = candidates[0]
        smallest_bucket_size = bucket_sizes[smallest_bucket]

        for cand in candidates:
            size = bucket_sizes[cand]
            if size < smallest_bucket_size:
                smallest_bucket = cand
                smallest_bucket_size = size
        
        # print(f"\rassigned vector {i+1} to {smallest_bucket}", end='', flush=True)
        index[rest_indexes[i]] = smallest_bucket
        bucket_sizes[smallest_bucket] +=1


def invert_index(index, B):
    inverted_index = [[] for _ in range(B)]
    for i, bucket in enumerate(index):
        inverted_index[bucket].append(i)
    return inverted_index

def build_index(train, config: Config):

    print("bulding index with preserved order...")
    logging.info(f"Started building index")
    SIZE, DIM = np.shape(train)

    sample_size = SIZE if SIZE < 2_000_00 else 1_000_000

    random_order = np.arange(SIZE)
    np.random.seed(42)
    np.random.shuffle(random_order)
    sample_indexes = np.sort(random_order[:sample_size])
    rest_indexes = np.sort(random_order[sample_size:])

    sample = train[sample_indexes, :]
    rest = train[rest_indexes, :] if rest_indexes.size > 0 else None

    print(f"sample size = {len(sample)}")

    save_dataset_as_memmap(train, config.dataset_name)

    print("finding neighbours...", flush=True)
    logging.info(f"Finding ground truths for train vectors")
    neighbours = get_train_nearest_neighbours_from_file(sample, config.nr_neighbours, sample_size, config.dataset_name)

    labels = []
    dataset = BLISSDataset(sample, labels, config.device)

    # final assignment arr for entire dataset
    final_bucket_assignments = np.empty(SIZE, dtype=np.uint32)

    final_index = []
    time_per_r = []
    process = psutil.Process(os.getpid())
    memory_usage = process.memory_info().rss / (1024 ** 2)
    log_mem(f"before_building_global={config.global_reass}_shuffle={config.shuffle}", memory_usage, config.memlog_path)
    for r in range(config.r):
        logging.info(f"Training model {r+1}")
        start = time.time()

        sample_buckets, bucket_sizes = assign_initial_buckets(sample_size, rest_indexes.size, r, config.b)
        logging.info(f"Random starting buckets initialized")
        # final_bucket_assignments[sample_indexes] = sample_buckets
        print("initial bucket sizes for training sample:")
        np.set_printoptions(threshold=6, suppress=True)
        print(bucket_sizes)

        print("making initial groundtruth labels", flush=True)
        labels = make_ground_truth_labels(config.b, neighbours, sample_buckets, sample_size, config.device)
        dataset.labels = labels

        print(f"setting up model {r+1}")
        torch.manual_seed(r)
        if config.device == torch.device("cuda"):
            torch.cuda.manual_seed(r)
        elif config.device == torch.device("mps"):
            torch.mps.manual_seed(r)
        model = BLISS_NN(DIM, config.b)
        print(f"training model {r+1}")
        train_model(model, dataset, sample_buckets, sample_size, bucket_sizes, neighbours, r, config)
        model_path = save_model(model, config.dataset_name, r+1, config.r, config.k, config.b, config.lr, config.shuffle, config.global_reass)
        print(f"model {r+1} saved to {model_path}.")

        final_bucket_assignments[sample_indexes] = sample_buckets

        if rest is not None:
            print("assigning remaining vectors to buckets...", flush=True)
            map_all_to_buckets(rest, rest_indexes, config.k, bucket_sizes, final_bucket_assignments, model_path, sample_size, DIM, config.b)
            # final_bucket_assignments[rest_indexes] = rest_buckets
        
        np.set_printoptions(threshold=np.inf, suppress=True)
        print(f"bucket_sizes sum = {np.sum(bucket_sizes)}", flush=True)
        print(bucket_sizes)
        inverted_index = invert_index(final_bucket_assignments, config.b)
        index_path = save_inverted_index(inverted_index, config.dataset_name, r+1, config.r, config.k, config.b, config.lr, config.shuffle, config.global_reass)
        final_index.append((index_path, model_path))
        end = time.time()
        time_per_r.append(end - start)

    build_time = time.time() - start
    process = psutil.Process(os.getpid())
    memory_usage = process.memory_info().rss / (1024 ** 2)
    log_mem(f"after_building_global={config.global_reass}_shuffle={config.shuffle}", memory_usage, config.memlog_path)
    return final_index, time_per_r, build_time, None

def load_model(model_path, dim, b):
    inf_device = torch.device("cpu")
    model = BLISS_NN(dim, b)
    model.load_state_dict(torch.load(model_path, weights_only=True, map_location=inf_device))
    model.eval()
    return model

def process_query_chunk(chunk, data, index, m, threshold, requested_amount):
    vectors, neighbours = chunk
    return query_multiple(data, index, vectors, neighbours, m, threshold, requested_amount, parallel=True)

def query_multiple_parallel(data, index, vectors, neighbours, m, threshold, requested_amount, num_workers):
    chunk_size = 200
    query_tasks = [(vectors[i:i+chunk_size], neighbours[i:i+chunk_size]) for i in range(0, len(vectors), chunk_size)]
    torch.set_num_threads(1)

    print(f"Number of query vectors: {len(vectors)}")
    print(f"Number of neighbour entries: {len(neighbours)}")
    print(f"Splitting queries into chunks of size {chunk_size} and dividing over {num_workers} processes", flush=True)

    try:
        process_func = partial(process_query_chunk, data=data, index=index, m=m, threshold=threshold, requested_amount=requested_amount)
        with Pool(processes=num_workers) as pool:
            results = pool.map(process_func, query_tasks)

        final_results = []
        for result in results:
            final_results.extend(result)

        print("\nQuerying completed.")
    except KeyboardInterrupt:
        print(f"\nQuerying process interrupted, exiting...")
        pool.terminate()
        pool.join()
        sys.exit(1)
    except Exception as e:
        print(f"\nAn error occurred during querying: {e}")
        pool.terminate()
        pool.join()
        sys.exit(1)

    return final_results

def query_multiple(data, index, vectors, neighbours, m, threshold, requested_amount, parallel=False):
    '''run multiple queries from a set of query vectors i.e. "Test" from the ANN benchmark datsets'''
    size = len(vectors)
    if not parallel:
        print(f"Number of query vectors: {size}")
        print(f"Number of neighbour entries: {len(neighbours)}", flush=True)
    query_times = []
    results = [[] for i in range(len(vectors))]
    for i, vector in enumerate(vectors):
        sys.stdout.write(f"\r[PID: {os.getpid()}] querying {i+1} of {size}       ")
        sys.stdout.flush()
        start = time.time()
        anns, dist_comps, recall = query(data, index, vector, neighbours[i], m, threshold, requested_amount)
        end = time.time()
        elapsed = end - start
        query_times.append(elapsed)
        results[i] = (anns, dist_comps, elapsed, recall)
    print("\r")
    return results

def query(data, index, query_vector, neighbours, m, freq_threshold, requested_amount):
    '''query the index for a single vector'''
    inverted_indexes, models = index
    candidates = {}
    for i in range(len(models)):
        model = models[i]
        model.eval()
        index = inverted_indexes[i]
        probabilities = None
        with torch.no_grad():
            probabilities = torch.sigmoid(model(query_vector))
        # print(probabilities)
        _, m_buckets = torch.topk(probabilities, m)
        m_buckets = m_buckets.tolist()
        seen = set()
        for bucket in m_buckets:
            for vector in index[bucket]:
                seen.add(vector)
        for vector in seen:
            f = 0
            if candidates.__contains__(vector):
                f = candidates.get(vector)
            candidates.update({vector: f+1}) 
    final_results = [key for key, value in candidates.items() if value >= freq_threshold]
    # print (f"final results = {len(final_results)}")
    if len(final_results) <= requested_amount:
        return final_results, 0, recall_single(final_results, neighbours)
    else:
        final_neighbours, dist_comps = reorder(data, query_vector, np.array(final_results, dtype=int), requested_amount)
        return final_neighbours, dist_comps, recall_single(final_neighbours, neighbours)
    
def reorder(data, query_vector, candidates, requested_amount): 
    n, d = np.shape(data)
    sp_index = []
    search_space = data[candidates]
    dist_comps = len(search_space)
    for i in range(len(search_space)):
        sp_index.append(candidates[i])
    # print(f"search_space = {search_space}")
    index = faiss.IndexFlatL2(d)
    index.add(search_space)
    query_vector = query_vector.reshape(1, -1)
    (dist, neighbours) = index.search(query_vector, requested_amount)
    neighbours = neighbours[0].tolist()
    final_neighbours = []
    for i in neighbours:
        final_neighbours.append(sp_index[i])
    return final_neighbours, dist_comps

def recall(results, neighbours):
    recalls = []
    for ann, nn in zip(results, neighbours):
        # get size of intersection of anns and nns to find the amount of correct anns
        correct = len(set(ann) & set(nn))
        recalls.append(correct/len(nn))
    return np.mean(recalls)

def recall_single(results, neighbours):
    return len(set(results) & set(neighbours))/len(neighbours)

def run_bliss(config: Config, mode, experiment_name):

    config.experiment_name = experiment_name
    dataset_name = config.dataset_name
    metric = dataset_name.split("-")[-1]
    print(f"Using device: {config.device}")
    print(dataset_name)

    if not os.path.exists(f"results/{experiment_name}/"):
        os.mkdir(f"results/{experiment_name}/")
    MEMLOG_PATH = f"results/{experiment_name}/{experiment_name}_memory_log.csv"
    config.memlog_path = MEMLOG_PATH

    inverted_indexes_paths = []
    if mode == 'build':
        data, _ = read_dataset(config.dataset_name, mode= 'train', size=config.datasize)
        print(f"shape memmap = {data.shape}")
        if metric == "angular":
            norms = np.linalg.norm(data, axis=1, keepdims=True)
            data = data / norms
        index, time_per_r, build_time, memory_usage = build_index(data, config)
        inverted_indexes_paths, model_paths = zip(*index)
        return time_per_r, build_time, memory_usage
    elif mode == 'query':
        logging.info("Loading models for inference")
        b = config.b if config.b !=0 else 1024
        config.b = b
        inverted_indexes_paths = []
        for i in range (config.r):
            inverted_indexes_paths.append(f"models/{dataset_name}_r{config.r}_k{config.k}_b{config.b}_lr{config.lr}_shf={config.shuffle}_gr={config.global_reass}/index_model{i+1}_{dataset_name}_r{i+1}_k{config.k}_b{config.b}_lr{config.lr}.pkl")
        model_paths = []
        for i in range(config.r):
            model_paths.append(f"models/{dataset_name}_r{config.r}_k{config.k}_b{config.b}_lr{config.lr}_shf={config.shuffle}_gr={config.global_reass}/model_{dataset_name}_r{i+1}_k{config.k}_b{config.b}_lr{config.lr}.pt")

        print(model_paths)
        print(inverted_indexes_paths)
        inverted_indexes = []
        for path in inverted_indexes_paths:
            with open(path, 'rb') as f:
                inverted_indexes.append(pickle.load(f))
        
        memmap_path = f"memmaps/memmap_{dataset_name}.npy"
        data = np.load(memmap_path, mmap_mode='r')

        size, dim = np.shape(data)

        q_models = [load_model(model_path, dim, b) for model_path in model_paths]
        index = (inverted_indexes, q_models)
        # print(index)
        
        logging.info("Reading query vectors and ground truths")
        test, neighbours = read_dataset(dataset_name, mode= 'test', size=config.datasize)
        # dataset = BigANNDataset(config.datasize)
        # dataset.prepare()
        # test = dataset.get_queries()
        # neighbours, _ = dataset.get_groundtruth()
        # print(neighbours.shape)
        if metric == "angular":
            norms = np.linalg.norm(test, axis=1, keepdims=True)
            test = test / norms

        print(f"creating tensor array from Test")
        test = torch.from_numpy(test).float()

        logging.info("Starting inference")
        num_workers = 16
        start = time.time()
        # results = query_multiple(data, index, test, neighbours, config.m, config.freq_threshold, config.nr_neighbours)
        results = query_multiple_parallel(data, index, test, neighbours, config.m, config.freq_threshold, config.nr_neighbours, num_workers)
        end = time.time()

        total_query_time = end - start

        anns = [t[0] for t in results]
        RECALL = recall(anns, neighbours)
        print(f"RECALL = {RECALL}", flush=True)

        return RECALL, results, total_query_time