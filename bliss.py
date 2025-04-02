import gc
import logging
import numpy as np
import os
import psutil  # type: ignore
import resource
import sys
import time
import torch
import tracemalloc
from pympler import asizeof
from torch.utils.data import DataLoader
from sklearn.utils import murmurhash3_32 as mmh3

import datasets as ds
import utils as ut
from bliss_model import BLISS_NN, BLISSDataset
from config import Config
from query import query_multiple, query_multiple_parallel, recall, load_data_for_inference
from train import train_model

    
def build_index(dataset: ds.Dataset, config: Config):
        
    print("bulding index...")
    logging.info(f"Started building index")
    SIZE = dataset.nb
    DIM = dataset.d
    # DIM = 64

    sample_size = SIZE if SIZE < 2_000_000 else 1_000_000

    memmap_path, mmp_shape = save_dataset_as_memmap(dataset, config, SIZE, DIM)
    
    sample = ut.get_training_sample_from_memmap(memmap_path, mmp_shape, sample_size, SIZE, DIM)
    print(f"sample size = {len(sample)}")

    print("finding neighbours...", flush=True)
    logging.info(f"Finding ground truths for train vectors")
    neighbours = ut.get_train_nearest_neighbours_from_file(sample, config.nr_train_neighbours, sample_size, config.dataset_name)

    labels = []
    dataset = BLISSDataset(sample, labels, config.device)

    final_index = []
    time_per_r = [] 
    process = psutil.Process(os.getpid())
    memory_usage = process.memory_info().rss / (1024 ** 2)
    bucket_size_stats = []
    tracemalloc.start()
    ut.log_mem(f"before_building_global={config.global_reass}_shuffle={config.shuffle}", memory_usage, config.memlog_path)
    for r in range(config.r):
        logging.info(f"Training model {r+1}")
        start = time.time()

        sample_buckets, bucket_sizes = assign_initial_buckets(sample_size, r, config.b)
        logging.info(f"Random starting buckets initialized")
        print("initial bucket sizes for training sample:")
        np.set_printoptions(threshold=6, suppress=True)
        print(bucket_sizes)

        print("making initial groundtruth labels", flush=True)
        labels = ut.make_ground_truth_labels(config.b, neighbours, sample_buckets, sample_size, config.device)
        dataset.labels = labels
        ds_size = asizeof.asizeof(dataset)
        ut.log_mem("size of training dataset before training (pq)", ds_size, config.memlog_path)

        print(f"setting up model {r+1}")
        ut.set_torch_seed(r, config.device)
        model = BLISS_NN(DIM, config.b)
        model_size = asizeof.asizeof(model)
        ut.log_mem("model size before training", model_size, config.memlog_path)
        print(f"training model {r+1}")
        train_model(model, dataset, sample_buckets, sample_size, bucket_sizes, neighbours, r, SIZE, config)
        current, _ = tracemalloc.get_traced_memory() 
        ut.log_mem(f"memory during training model {r+1}", current, config.memlog_path)
        model_path = ut.save_model(model, config.dataset_name, r+1, config.r, config.k, config.b, config.lr, config.shuffle, config.global_reass)
        print(f"model {r+1} saved to {model_path}.")

        model.eval()
        model.to(config.device)
        index = None
        if sample_size < SIZE:
            index = build_full_index(bucket_sizes, SIZE, model, config)
        else:
            index = sample_buckets
        del model 
        inverted_index, offsets = invert_index(index, bucket_sizes, SIZE)
        index_path = ut.save_inverted_index(inverted_index, offsets, config.dataset_name, r+1, config.r, config.k, config.b, config.lr, config.shuffle, config.global_reass)
        del inverted_index, offsets
        final_index.append((index_path, model_path))
        end = time.time()
        time_per_r.append(end - start)
        bucket_size_stats.append(bucket_sizes)

    build_time = time.time() - start
    # process = psutil.Process(os.getpid())
    # memory_usage = process.memory_info().rss / (1024 ** 2)
    _, peak_mem = tracemalloc.get_traced_memory()
    load_balance = ut.calc_load_balance(bucket_size_stats)
    tracemalloc.stop()
    ut.log_mem(f"after_building_global={config.global_reass}_shuffle={config.shuffle}", memory_usage, config.memlog_path)
    return final_index, time_per_r, build_time, peak_mem, load_balance

def assign_initial_buckets(train_size, r, B):
    '''
    assign bucket labels to vectors (indeces in the nd.array) using a hash function.
    the hash fucntion used here is the same as in the original code from the BLISS github.
    TODO: add reference link
    '''
    index = np.zeros(train_size, dtype=np.uint32) # from 0 to train_size-1
    bucket_sizes = np.zeros(B, dtype=np.uint32)

    for i in range(train_size):
        bucket = mmh3(i,seed=r)%B
        index[i] = bucket
        bucket_sizes[bucket] += 1
    
    return np.array(index, dtype=np.uint32), bucket_sizes

def map_all_to_buckets(map_loader, k, index, bucket_sizes, map_model, offset, device):
    '''
    Do a forward pass on the model with batches of vectors, then assign the vectors to a bucket one by one.
    '''
    logging.info(f"Mapping all train vectors to buckets")
    with torch.no_grad():
        for batch_data, _, batch_indices in map_loader:
            candidate_buckets = ut.get_topk_buckets_for_batch(batch_data, k, map_model, device).numpy()
            
            for i, item_index in enumerate(batch_indices):
                item_idx = item_idx + offset
                ut.reassign_vector_to_bucket(index, bucket_sizes, candidate_buckets, i, item_index)

def map_all_to_buckets_base(index, bucket_sizes, full_data, data_batched, data_loader, N, k, model, device):
    candidate_buckets = np.zeros(shape= (N, k)) # all topk buckets per vector shape = (N, k).
    offset = 0
    for batch in full_data.get_dataset_iterator(bs = 1_000_00):
        data_batched.data = batch
        ut.get_all_topk_buckets(data_loader, k, candidate_buckets, model, offset, device)
        offset += len(batch)

    for i in range(N):
        ut.reassign_vector_to_bucket(index, bucket_sizes, candidate_buckets, i, i)

def build_full_index(bucket_sizes, SIZE, model, config: Config):
    bucket_sizes[:] = 0 
    index = np.zeros(SIZE, dtype=int)
    full_data = ut.get_dataset_obj(config.dataset_name, config.datasize)
    data_batched = BLISSDataset(None, None, device = torch.device("cpu"), mode='build')
    map_loader = DataLoader(data_batched, batch_size=config.batch_size, shuffle=False, num_workers=8)
    # map all vectors to buckets using the chosen reassignment strategy
    start = time.time()
    if config.reass_mode == 0: # baseline implementation 
        map_all_to_buckets_base(index, bucket_sizes, full_data, data_batched, map_loader, SIZE, config.k, model, config.device)

    elif config.reass_mode == 1: # reassign all vectors in a foward pass batch directly 
        global_idx = 0
        for batch in full_data.get_dataset_iterator(bs=1_000_000):
            data_batched.data = batch
            map_all_to_buckets(map_loader, config.k, index, bucket_sizes, model, global_idx)
            global_idx += len(batch)

    elif config.reass_mode == 2:
        # Do all forward passes sequentially and then do reassignments in batches
        candidate_buckets = np.zeros(shape= (SIZE, config.k)) # all topk buckets per vector shape = (N, k).
        offset = 0
        for batch in full_data.get_dataset_iterator(bs = 1_000_00):
            data_batched.data = batch
            ut.get_all_topk_buckets(map_loader, config.k, candidate_buckets, model, offset, config.device)
            offset += len(batch)

        chunk_size = 1024
        for i in range(start=0, stop=SIZE, step=chunk_size):
            # get the topk buckets per vector
            topk_per_vector = candidate_buckets[i : min(i + chunk_size, SIZE)] # shape = (chunk_size, k)
            # get sizes of cnadidates per vector
            candidate_sizes_per_vector = bucket_sizes[topk_per_vector]
            # get the least ocupied of each candidate set 
            least_occupied = topk_per_vector[np.argmin(candidate_sizes_per_vector, axis=1)] # shape = (chunk_size, 1)
            index[i : min(i + chunk_size, SIZE)] = least_occupied

            bucket_increments = np.bincount(least_occupied, minlength=len(bucket_sizes))
            print(bucket_increments)
            bucket_sizes = np.add(bucket_sizes, bucket_increments)

    elif config.reass_mode == 3:
        # Alternate fowardpasses wih batched reassignment -> vectorised assignment of a whole batch of buckets at once
        pass

    end = time.time()
    logging.info(f"final assignment mode:{config.reass_mode} took {end-start} seconds")

    return index 

def invert_index(index, bucket_sizes, SIZE):
    '''
    Given an index, build an inverted index. The inverted index consists of an inverted index file and offset file.
    The inverted index file contains the indices of the vectors, ordered by bucket.
    The offsets file contains the cumulative sum of the amount of vectors in the buckets, so they can be used to identify the start and end
    of each bucket in the inverted index.
    '''
    offsets = np.cumsum(bucket_sizes)
    inverted_index = np.zeros(SIZE, dtype=np.uint32)
    inverted_index = np.argsort(index)
    np.set_printoptions(threshold=np.inf, suppress=True)
    print(f"bucket_sizes sum = {np.sum(bucket_sizes)}", flush=True)
    print(bucket_sizes)
    return inverted_index, offsets

def load_indexes_and_models(config: Config, SIZE, DIM, b):
    inverted_indexes_paths = []
    offsets_paths = []
    model_paths = []
    for i in range (config.r):
        inverted_indexes_paths.append(f"models/{config.dataset_name}_r{config.r}_k{config.k}_b{config.b}_lr{config.lr}_shf={config.shuffle}_reass={config.reass_mode}/index_model{i+1}_{config.dataset_name}_r{i+1}_k{config.k}_b{config.b}_lr{config.lr}.npy")
        offsets_paths.append(f"models/{config.dataset_name}_r{config.r}_k{config.k}_b{config.b}_lr{config.lr}_shf={config.shuffle}_reass={config.reass_mode}/offsets_model{i+1}_{config.dataset_name}_r{i+1}_k{config.k}_b{config.b}_lr{config.lr}.npy")
        model_paths.append(f"models/{config.dataset_name}_r{config.r}_k{config.k}_b{config.b}_lr{config.lr}_shf={config.shuffle}_reass={config.reass_mode}/model_{config.dataset_name}_r{i+1}_k{config.k}_b{config.b}_lr{config.lr}.pt")

    indexes = np.zeros(shape = (config.r, SIZE), dtype=np.uint32)
    offsets = np.zeros(shape = (config.r, config.b), dtype=np.uint32)
    for i, (inv_path, off_path) in enumerate(zip(inverted_indexes_paths, offsets_paths)):
        inds_load = np.load(inv_path)
        ind = np.array(inds_load)
        indexes[i] = ind
        offs_load = np.load(off_path)
        off = np.array(offs_load)
        offsets[i] = off
    
    # load models and indexes into memory
    q_models = [ut.load_model(model_path, DIM, b) for model_path in model_paths]
    index = ((indexes, offsets), q_models)
    return index

def save_dataset_as_memmap(dataset, config: Config, SIZE, DIM):
    '''
    Put a dataset into a memmap and return the path where it was saved. 
    Small datasets can be loaded into memory and written to a memmap in one go, larger datasets are processed in chunks.
    '''
    memmap_path = f"memmaps/{config.dataset_name}_{config.datasize}.npy"
    mmp_shape = (SIZE, DIM)
    print(f"mmp shape = {mmp_shape}")
    mmp = None
    if not os.path.exists(memmap_path):
        mmp = np.memmap(memmap_path, mode ="w+", shape=mmp_shape, dtype=np.float32)
        if SIZE >= 10_000_000:
            fill_memmap_in_batches(dataset, config, mmp)
        else:
            data = dataset.get_dataset()[:]
            if dataset.distance() == "angular":
                data = ut.normalise_data(data)
            data = ut.random_projection(data, DIM)
            mmp[:] = data
            mmp.flush()
    return memmap_path, mmp_shape

def fill_memmap_in_batches(dataset, config: Config, mmp):
    '''
    Save the dataset in a memmap in batches if the dataset is too large to load into memory in one go.
    '''
    index = 0
    for batch in dataset.get_dataset_iterator(1_000_000):
        process = psutil.Process(os.getpid())
        memory_usage = process.memory_info().rss / (1024 ** 2)
        ut.log_mem(f"while loading for memmap batch: ", memory_usage, config.memlog_path)
        print(f"mem usage while loading batch: {memory_usage}")
        batch_size = len(batch)
        mmp[index: index + batch_size] = batch
        mmp.flush()
        del batch
        gc.collect()
        index += batch_size

def run_bliss(config: Config, mode, experiment_name):
    '''
    Run the BLISS algorithm. Mode determines whether an index is built or whether inference is run on an existing index.
    '''

    config.experiment_name = experiment_name
    print(f"Using device: {config.device}")
    print(config.dataset_name)

    if not os.path.exists(f"results/{experiment_name}/"):
        os.mkdir(f"results/{experiment_name}/")
    MEMLOG_PATH = f"results/{experiment_name}/{experiment_name}_memory_log.csv"
    config.memlog_path = MEMLOG_PATH

    dataset = ut.get_dataset_obj(config.dataset_name, config.datasize)
    SIZE = dataset.nb
    DIM = dataset.d
    dataset.prepare()
    if mode == 'build':
        index, time_per_r, build_time, memory_usage, load_balance = build_index(dataset, config)
        usage = resource.getrusage(resource.RUSAGE_SELF)
        peak_mem = usage.ru_maxrss / 1_000_000 if sys.platform == 'darwin' else usage.ru_maxrss / 1000
        ut.log_mem("peak_mem_building", peak_mem, MEMLOG_PATH)
        return time_per_r, build_time, memory_usage, load_balance
    elif mode == 'query':
        # set b if it wasn't already set in config
        b = config.b if config.b !=0 else 1024
        config.b = b
        
        logging.info("Loading models for inference")
        index = load_indexes_and_models(config, SIZE, DIM, b)
        logging.info("Reading query vectors and ground truths")
        tracemalloc.start()
        data, test, neighbours = load_data_for_inference(dataset, config, SIZE, DIM)
        current, peak = tracemalloc.get_traced_memory()
        ut.log_mem(f"memory afer loading memmap for inference", current, config.memlog_path)
        ut.log_mem(f"memory peak after loading memmap for inference", peak, config.memlog_path)
        tracemalloc.stop()

        print(f"creating tensor array from Test")
        test = torch.from_numpy(test).to(torch.float32)

        logging.info("Starting inference")
        num_workers = 8
        start = time.time()
        tracemalloc.start()
        results = query_multiple(data, index, test, neighbours, config.m, config.freq_threshold, config.nr_ann)
        # results = query_multiple_parallel(data, index, test, neighbours, config.m, config.freq_threshold, config.nr_ann, num_workers)
        current, peak  = tracemalloc.get_traced_memory()
        ut.log_mem("peak memory during querying", peak , config.memlog_path)
        tracemalloc.stop()
        end = time.time()

        total_query_time = end - start

        anns = [t[0] for t in results]
        RECALL = recall(anns, neighbours)
        print(f"RECALL = {RECALL}", flush=True)
        usage = resource.getrusage(resource.RUSAGE_SELF)
        peak_mem = usage.ru_maxrss / 1_000_000 if sys.platform == 'darwin' else usage.ru_maxrss / 1000
        ut.log_mem("peak_mem_querying", peak_mem, MEMLOG_PATH)
        return RECALL, results, total_query_time