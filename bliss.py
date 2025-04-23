import faiss
import gc
import logging
import numpy as np
import optuna
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
from query import query_multiple, query_multiple_batched, load_data_for_inference
from train import train_model
    
def build_index(dataset: ds.Dataset, config: Config, trial=None):
        
    print("bulding index...")
    logging.info(f"Started building index")
    SIZE = dataset.nb
    DIM = dataset.d
    # DIM = 64

    sample_size = SIZE 
    if SIZE == 10_000_000:
        sample_size = 1_000_000
    if SIZE == 100_000_000:
        sample_size = 2_000_000

    memmap_path, mmp_shape = save_dataset_as_memmap(dataset, config, SIZE, DIM)
    
    sample = ut.get_training_sample_from_memmap(memmap_path, mmp_shape, sample_size, SIZE, DIM)
    print(f"sample size = {len(sample)}")
    sample_mem_size = asizeof.asizeof(sample)
    ut.log_mem("memory of train sample object", sample_mem_size, config.memlog_path)

    print("finding neighbours...", flush=True)
    neighbours = ut.get_train_nearest_neighbours_from_file(sample, config.nr_train_neighbours, sample_size, config.dataset_name, config.datasize)
    neighbours_mem_size = asizeof.asizeof(neighbours)
    ut.log_mem("memory of neighbours object", neighbours_mem_size, config.memlog_path)

    # labels = torch.zeros((1, 1))
    dataset = BLISSDataset(sample, config.device)

    final_index = []
    time_per_r = [] 
    memory_training = 0
    memory_final_assignment = 0
    bucket_size_stats = []
    model_sizes_total = 0
    index_sizes_total = 0
    for r in range(config.r):
        logging.info(f"Training model {r+1}")
        start = time.time()

        sample_buckets, bucket_sizes = assign_initial_buckets(sample_size, r, config.b)
        logging.info(f"Random starting buckets initialized")
        print("initial bucket sizes for training sample:")
        np.set_printoptions(threshold=6, suppress=True)
        print(bucket_sizes)

        print("making initial groundtruth labels", flush=True)
        # labels = ut.make_ground_truth_labels(config.b, neighbours, sample_buckets, sample_size)
        # dataset.labels = labels
        ds_size = asizeof.asizeof(dataset)
        ut.log_mem("size of training dataset before training", ds_size, config.memlog_path)
        
        print(f"setting up model {r+1}")
        ut.set_torch_seed(r, config.device)
        model = BLISS_NN(DIM, config.b)
        model_size = asizeof.asizeof(model)
        ut.log_mem("model size before training", model_size, config.memlog_path)
        print(f"training model {r+1}")
        memory_training_current = train_model(model, dataset, sample_buckets, sample_size, bucket_sizes, neighbours, r, SIZE, config)
        ut.log_mem(f"memory during training model {r+1}", memory_training_current, config.memlog_path)
        memory_training = memory_training_current if memory_training_current > memory_training else memory_training
        model_path, model_file_size = ut.save_model(model, config.dataset_name, r+1, config.r, config.k, config.b, config.lr, config.batch_size, config.reass_mode)
        model_sizes_total += model_file_size
        print(f"model {r+1} saved to {model_path}.")

        #prune trial when buckets are too unbalanced ie. normalised entropy of bucketsizes is too low
        if trial is not None:
            ne = ut.norm_ent(bucket_sizes) # get normalised entropy for current bucketsizes
            trial.report(ne, step=1)
            threshold = 0.7
            if ne < threshold:
                raise optuna.exceptions.TrialPruned(f"buckets too unbalanced normalised entropy = {ne}, min is {threshold}")
        
        model.eval()
        model.to(config.device)
        index = None
        if sample_size < SIZE:
            index, memory_final_assignment = build_full_index(bucket_sizes, SIZE, model, config)
        else:
            index = sample_buckets
        del model 
            
        inverted_index, offsets = invert_index(index, bucket_sizes, SIZE)
        index_path, index_files_size = ut.save_inverted_index(inverted_index, offsets, config.dataset_name, r+1, config.r, config.k, config.b, config.lr, config.batch_size, config.reass_mode)
        index_sizes_total += index_files_size
        del inverted_index, offsets
        final_index.append((index_path, model_path))
        end = time.time()
        time_per_r.append(end - start)
        bucket_size_stats.append(bucket_sizes)

    build_time = time.time() - start
    load_balance = ut.calc_load_balance(bucket_size_stats)

    ut.log_mem(f"final_reass={config.reass_mode}_reass_chunk_size={config.reass_chunk_size}", memory_final_assignment, config.memlog_path)
    ut.log_mem(f"training={config.reass_mode}_batch_size={config.batch_size}", memory_training, config.memlog_path)
    return final_index, time_per_r, build_time, memory_final_assignment, memory_training, load_balance, index_sizes_total, model_sizes_total

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

def map_all_to_buckets_1(map_loader, full_data, data_batched, k, index, bucket_sizes, map_model, device):
    '''
    Do a forward pass on the model with batches of vectors, then assign the vectors to a bucket one by one.
    '''
    logging.info(f"Mapping all train vectors to buckets")
    offset = 0
    process = psutil.Process(os.getpid())
    memory_usage = 0
    for batch in full_data.get_dataset_iterator(bs=1_000_000):
        data_batched.data = torch.from_numpy(batch).float()
        with torch.no_grad():
            for batch_data, batch_indices in map_loader:
                candidate_buckets = ut.get_topk_buckets_for_batch(batch_data, k, map_model, device).numpy()
                mem_current = process.memory_full_info().uss / (1024 ** 2)
                memory_usage = mem_current if mem_current>memory_usage else memory_usage
                for i, item_index in enumerate(batch_indices):
                    item_idx = item_index + offset
                    ut.reassign_vector_to_bucket(index, bucket_sizes, candidate_buckets, i, item_idx)
    
        offset += len(batch)
    return memory_usage

def map_all_to_buckets_0(index, bucket_sizes, full_data, data_batched, data_loader, N, k, model, device):
    '''
    Baseline implementation using the same strategy as the BLISS original code <TODO-insert github link>.
    Gets all candidate buckets per vector at once and does the reassignment sequentially.
    '''
    print("started map all", flush=True)
    process = psutil.Process(os.getpid())
    candidate_buckets = get_all_candidate_buckets(len(index), model, k, device, full_data, data_batched, data_loader)
    memory_usage = process.memory_full_info().uss / (1024 ** 2)
    print("finished getting topk", flush=True)

    for i in range(N):
        ut.reassign_vector_to_bucket(index, bucket_sizes, candidate_buckets, i, i)
    return memory_usage

def build_full_index(bucket_sizes, SIZE, model, config: Config):
    bucket_sizes[:] = 0 
    memory_usage = 0
    reass_mode, k, device = config.reass_mode, config.k, config.device
    index = np.zeros(SIZE, dtype=np.uint32)
    full_data = ut.get_dataset_obj(config.dataset_name, config.datasize)
    data_batched = BLISSDataset(None, device = torch.device("cpu"), mode='build')
    chunk_size = config.reass_chunk_size
    map_loader = DataLoader(data_batched, batch_size=chunk_size, shuffle=False, num_workers=8)
    # map all vectors to buckets using the chosen reassignment strategy
    start = time.time()
    if reass_mode == 0: # baseline implementation 
        memory_usage = map_all_to_buckets_0(index, bucket_sizes, full_data, data_batched, map_loader, SIZE, k, model, device)

    elif reass_mode == 1: # reassign all vectors in a foward pass batch directly 
        memory_usage = map_all_to_buckets_1(map_loader, full_data, data_batched, k, index, bucket_sizes, model, device)

    elif reass_mode == 2:
        # Do all forward passes sequentially and then do reassignments in batches
        candidate_buckets = get_all_candidate_buckets(SIZE, model, k, device, full_data, data_batched, map_loader)

        for i in range(0, SIZE, chunk_size):
            # get the topk buckets per vector
            topk_per_vector = candidate_buckets[i : min(i + chunk_size, SIZE)] # shape = (chunk_size, k)
            memory_current = assign_to_buckets_vectorised(bucket_sizes, SIZE, index, chunk_size, i, topk_per_vector)
            memory_usage = memory_current if memory_current>memory_usage else memory_usage

    elif reass_mode == 3:
        # Alternate fowardpasses wih batched reassignment -> vectorised assignment of a whole batch of buckets at once
        logging.info(f"Mapping all to buckets mode: {reass_mode}")
        offset = 0
        process = psutil.Process(os.getpid())
        for batch in full_data.get_dataset_iterator(bs=1_000_000):
            data_batched.data = torch.from_numpy(batch).float()
            with torch.no_grad():
                for batch_data, _ in map_loader:
                    topk_per_vector = ut.get_topk_buckets_for_batch(batch_data, k, model, config.device).numpy()
                    memory_current = assign_to_buckets_vectorised(bucket_sizes, SIZE, index, chunk_size, offset, topk_per_vector)
                    memory_current = process.memory_full_info().uss / (1024 ** 2)
                    memory_usage = memory_usage if memory_current > memory_usage else memory_usage
                    offset += chunk_size

    end = time.time()
    logging.info(f"final assignment mode:{reass_mode} took {end-start} seconds")

    return index, memory_usage

def assign_to_buckets_vectorised(bucket_sizes, SIZE, index, chunk_size, i, topk_per_vector):
    process = psutil.Process(os.getpid())
    candidate_sizes_per_vector = bucket_sizes[topk_per_vector]    
    vectors = np.arange(topk_per_vector.shape[0])
    sizes = np.argmin(candidate_sizes_per_vector, axis=1)
    # get the least ocupied of each candidate set 
    least_occupied = topk_per_vector[vectors, sizes]
    memory_usage = process.memory_full_info().uss / (1024 ** 2)
    index[i : min(i + chunk_size, SIZE)] = least_occupied

    bucket_increments = np.bincount(least_occupied, minlength=len(bucket_sizes))
    bucket_sizes[:] = np.add(bucket_sizes, bucket_increments)
    return memory_usage

def get_all_candidate_buckets(SIZE, model, k, device, full_data, data_batched, map_loader):
    candidate_buckets = np.zeros(shape= (SIZE, k), dtype=np.uint32) # all topk buckets per vector shape = (N, k).
    offset = 0
    for batch in full_data.get_dataset_iterator(bs = 1_000_00):
        data_batched.data = torch.from_numpy(batch).float()
        ut.get_all_topk_buckets(map_loader, k, candidate_buckets, model, offset, device)
        offset += len(batch)
    return candidate_buckets 

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
        inverted_indexes_paths.append(f"models/{config.dataset_name}_r{config.r}_k{config.k}_b{config.b}_lr{config.lr}_bs={config.batch_size}_reass={config.reass_mode}/index_model{i+1}_{config.dataset_name}_r{i+1}_k{config.k}_b{config.b}_lr{config.lr}.npy")
        offsets_paths.append(f"models/{config.dataset_name}_r{config.r}_k{config.k}_b{config.b}_lr{config.lr}_bs={config.batch_size}_reass={config.reass_mode}/offsets_model{i+1}_{config.dataset_name}_r{i+1}_k{config.k}_b{config.b}_lr{config.lr}.npy")
        model_paths.append(f"models/{config.dataset_name}_r{config.r}_k{config.k}_b{config.b}_lr{config.lr}_bs={config.batch_size}_reass={config.reass_mode}/model_{config.dataset_name}_r{i+1}_k{config.k}_b{config.b}_lr{config.lr}.pt")

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
    logging.info("Creating dataset memmap")
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
            mmp[:] = data
            mmp.flush()
    del mmp
    return memmap_path, mmp_shape

def fill_memmap_in_batches(dataset, config: Config, mmp):
    '''
    Save the dataset in a memmap in batches if the dataset is too large to load into memory in one go.
    '''
    index = 0
    for batch in dataset.get_dataset_iterator(bs=1_000_000):
        process = psutil.Process(os.getpid())
        memory_usage = process.memory_full_info().uss / (1024 ** 2)
        ut.log_mem(f"while loading for memmap batch: ", memory_usage, config.memlog_path)
        print(f"mem usage while loading batch: {memory_usage}")
        batch_size = len(batch)
        mmp[index: index + batch_size] = batch
        mmp.flush()
        del batch
        gc.collect()
        index += batch_size

def run_bliss(config: Config, mode, experiment_name, trial=None):
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
        index, time_per_r, build_time, memory_final_assignment, memory_training, load_balance, index_sizes_total, model_sizes_total = build_index(dataset, config, trial)
        usage = resource.getrusage(resource.RUSAGE_SELF)
        peak_mem = usage.ru_maxrss / 1_000_000 if sys.platform == 'darwin' else usage.ru_maxrss / 1000
        ut.log_mem("peak_mem_building", peak_mem, MEMLOG_PATH)
        return time_per_r, build_time, memory_final_assignment, memory_training, load_balance, index_sizes_total, model_sizes_total
    elif mode == 'query':
        # set b if it wasn't already set in config
        b = config.b if config.b !=0 else 1024
        config.b = b
        
        logging.info("Loading models for inference")
        index = load_indexes_and_models(config, SIZE, DIM, b)
        logging.info("Reading query vectors and ground truths")
        process = psutil.Process(os.getpid())
        data, test, neighbours = load_data_for_inference(dataset, config, SIZE, DIM)
        mem_load = process.memory_full_info().uss / (1024**2)
        ut.log_mem("memory after loading data", mem_load, MEMLOG_PATH)
 
        print(f"creating tensor array from Test")
        test = torch.from_numpy(test).to(torch.float32)
        data_pq = None
        if config.pq:
            start = time.time()
            sample, _ = ut.get_training_sample(dataset, 1_000_000, SIZE, DIM)
            data_pq, _ = ut.train_pq(sample.astype(np.float32), 32, 8)
            del sample
            data_pq.add(data)
            del data
            faiss.write_index(data_pq, 'datapq.index')
            pq_index_size = os.path.getsize('datapq.index')
            ut.log_mem(f"size of data (pq index)", pq_index_size, MEMLOG_PATH)
            print(f"preprocessing for pq took {time.time()-start}")
        else:
            index_size = data.nbytes
            ut.log_mem("size of data", index_size, MEMLOG_PATH)
        
        logging.info("Starting inference")
        start = time.time()
        # results = query_multiple(data, index, test, neighbours, config.m, config.freq_threshold, config.nr_ann)
        mem_pre_query = process.memory_full_info().uss / (1024 ** 2)
        ut.log_mem(f"memory before querying pq:{config.pq}", mem_pre_query, MEMLOG_PATH)
        qstart = time.time()
        if config.pq:
            if config.query_batched:
                results = query_multiple_batched(data_pq, index, test, neighbours, config)
            else:
                results = query_multiple(data_pq, index, test, neighbours, config)
        else:
            if config.query_batched:
                results = query_multiple_batched(data, index, test, neighbours, config)
            else:
                results = query_multiple(data, index, test, neighbours, config)
        mem_post_query = process.memory_full_info().uss / (1024 ** 2)
        ut.log_mem("peak memory during querying", mem_post_query , config.memlog_path)
        print(f"querying pq={config.pq} took {time.time()-qstart}")
        end = time.time()

        total_query_time = end - start

        anns = [t[0] for t in results]
        RECALL = ut.recall(anns, neighbours)
        print(f"RECALL = {RECALL}", flush=True)
        return RECALL, results, total_query_time
    