import gc
import logging
import numpy as np
import optuna
import os
import psutil  # type: ignore
import time
import torch
from torch.utils.data import DataLoader
from sklearn.utils import murmurhash3_32 as mmh3

import datasets as ds
import utils as ut
from bliss_model import BLISS_NN, BLISSDataset
from config import Config
from query import query_multiple, query_multiple_batched, load_data_for_inference, query_multiple_batched_twostep, query_multiple_twostep
from train import train_model
    
def build_index(dataset: ds.Dataset, config: Config, trial=None):
    '''
    Main index build loop
    '''
        
    print("bulding index...")
    logging.info(f"Started building index")
    SIZE = dataset.nb
    DIM = dataset.d

    sample_size = SIZE 
    if SIZE == 10_000_000 or SIZE == 100_000_000:
        sample_size = 1_000_000
    elif SIZE == 1_000_000_000:
        sample_size = 10_000_000
    
    sample = ut.get_training_sample_from_memmap(dataset, sample_size, SIZE, DIM, config.dataset_name, config.datasize)
    print(f"sample size = {len(sample)}")

    print("finding neighbours...", flush=True)
    neighbours = ut.get_train_nearest_neighbours_from_file(sample, config.nr_train_neighbours, sample_size, config.dataset_name, config.datasize)

    dataset = BLISSDataset(sample, config.device)

    final_index = []
    train_time_per_r = [] 
    final_assign_time_per_r = []
    load_balances_all = []
    memory_training = 0
    memory_final_assignment = 0
    bucket_size_stats = []
    model_sizes_total = 0
    index_sizes_total = 0
    start = time.time()
    for r in range(config.r):
        logging.info(f"Training model {r+1}")
        start_training = time.time()

        sample_buckets, bucket_sizes = assign_initial_buckets(sample_size, r, config.b)
        logging.info(f"Random starting buckets initialized")
        print("initial bucket sizes for training sample:")
        np.set_printoptions(threshold=6, suppress=True)
        print(bucket_sizes)

        print("making initial groundtruth labels", flush=True)
        
        print(f"setting up model {r+1}")
        ut.set_torch_seed(r, config.device)
        model = BLISS_NN(DIM, config.b)
        print(f"training model {r+1}")
        memory_training_current, load_balances = train_model(model, dataset, sample_buckets, sample_size, bucket_sizes, neighbours, r, SIZE, config)
        memory_training = memory_training_current if memory_training_current > memory_training else memory_training
        model_path, model_file_size = ut.save_model(model, config.dataset_name, r+1, config.r, config.k, config.b, config.lr, config.batch_size, config.reass_mode, config.reass_chunk_size, config.epochs, config.iterations)
        train_time_per_r.append(time.time() - start_training)
        model_sizes_total += model_file_size
        print(f"model {r+1} saved to {model_path}.")

        # FOR OPTUNA: 
        # prune trial when buckets are too unbalanced ie. normalised entropy of bucketsizes is too low
        if trial is not None:
            ne = ut.norm_ent(bucket_sizes) # get normalised entropy for current bucketsizes
            trial.report(ne, step=1)
            threshold = 0.8
            if ne < threshold:
                raise optuna.exceptions.TrialPruned(f"buckets too unbalanced normalised entropy = {ne}, min is {threshold}")
        
        model.eval()
        model.to(config.device)
        index = None
        if sample_size < SIZE:
            final_assign_start = time.time()
            index, memory_final_assignment = build_full_index(bucket_sizes, SIZE, model, config)
            final_assign_time_per_r.append(time.time()-final_assign_start)
        else:
            index = sample_buckets
            final_assign_time_per_r.append(0)
        del model
        inverted_index, offsets = invert_index(index, bucket_sizes, SIZE)
        index_path, index_files_size = ut.save_inverted_index(inverted_index, offsets, config.dataset_name, r+1, config.r, config.k, config.b, config.lr, config.batch_size, config.reass_mode, config.reass_chunk_size, config.epochs, config.iterations)
        # TODO: write RP files if query_twostep is enabled!
        index_sizes_total += index_files_size
        del inverted_index, offsets
        final_index.append((index_path, model_path))
        bucket_size_stats.append(bucket_sizes)
        load_balances_all.append(load_balances)

    build_time = time.time() - start
    normalised_entropy = ut.norm_ent(bucket_sizes)

    return final_index, sum(train_time_per_r), sum(final_assign_time_per_r), build_time, memory_final_assignment, memory_training, normalised_entropy, index_sizes_total, model_sizes_total, load_balances_all

def assign_initial_buckets(train_size, r, B):
    '''
    assign bucket labels to vectors (indeces in the nd.array) using a hash function.
    the hash fucntion used here is the same as in the original code from the BLISS github.
    TODO: add reference link
    '''
    index = np.zeros(train_size, dtype=np.int32) # from 0 to train_size-1
    bucket_sizes = np.zeros(B, dtype=np.int32)

    for i in range(train_size):
        bucket = mmh3(i,seed=r)%B
        index[i] = bucket
        bucket_sizes[bucket] += 1
    
    return np.array(index, dtype=np.int32), bucket_sizes

def map_all_to_buckets_0(index, bucket_sizes, full_data, data_batched, data_loader, N, k, model, device, mem_tracking=False):
    '''
    Baseline implementation using the same strategy as the BLISS original code (https://github.com/gaurav16gupta/BLISS).
    Gets all candidate buckets per vector at once and does the reassignment sequentially.
    '''
    print("started map all", flush=True)
    memory_usage = 0
    process = psutil.Process(os.getpid())
    candidate_buckets, vram  = get_all_candidate_buckets(len(index), model, k, device, full_data, data_batched, data_loader, mem_tracking)
    if mem_tracking:
        memory_usage = process.memory_full_info().uss / (1024 ** 2)
        memory_usage = vram if vram > memory_usage else memory_usage
    print("finished getting topk", flush=True)

    for i in range(N):
        ut.reassign_vector_to_bucket(index, bucket_sizes, candidate_buckets[i], i)
    return memory_usage

def map_all_to_buckets_1(map_loader, full_data, data_batched, k, index, bucket_sizes, map_model, device, mem_tracking=False):
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
                candidate_buckets, vram = ut.get_topk_buckets_for_batch(batch_data, k, map_model, device)
                candidate_buckets.numpy()
                if mem_tracking:
                    mem_current = process.memory_full_info().uss / (1024 ** 2)
                    memory_usage = mem_current if mem_current > memory_usage else memory_usage
                    memory_usage = vram if vram > memory_usage else memory_usage
                for i, item_index in enumerate(batch_indices):
                    item_idx = item_index + offset
                    ut.reassign_vector_to_bucket(index, bucket_sizes, candidate_buckets[i], item_idx)
    
        offset += len(batch)
    return memory_usage

def map_all_to_buckets_2(bucket_sizes, SIZE, model, memory_usage, k, device, index, full_data, data_batched, chunk_size, map_loader, mem_tracking=False):
    '''
    First collect the top-k buckets for all vectors by doing forward passes on the model. 
    Then go through the vectors in batches, reassigning a whole batch at once.
    '''
    candidate_buckets, vram = get_all_candidate_buckets(SIZE, model, k, device, full_data, data_batched, map_loader)
    memory_usage=0
    for i in range(0, SIZE, chunk_size):
        # get the topk buckets per vector
        topk_per_vector = candidate_buckets[i : min(i + chunk_size, SIZE)] # shape = (chunk_size, k)
        if mem_tracking:
            memory_current = ut.assign_to_buckets_vectorised(bucket_sizes, SIZE, index, chunk_size, i, topk_per_vector)
            memory_usage = memory_current if memory_current>memory_usage else memory_usage
            memory_usage = vram if vram > memory_usage else memory_usage
    return memory_usage

def map_all_to_buckets_3(bucket_sizes, SIZE, model, config, memory_usage, reass_mode, k, index, full_data, data_batched, chunk_size, map_loader, mem_tracking=False):
    '''
    Go through the data in batches. For each batch, get the predicted top-k buckets for all vectors by doing a forward pass on the model.
    Then reassign the whole batch.
    '''
    logging.info(f"Mapping all to buckets mode: {reass_mode}")
    offset = 0
    memory_usage=0
    for batch in full_data.get_dataset_iterator(bs=1_000_000):
        data_batched.data = torch.from_numpy(batch).float()
        with torch.no_grad():
            for batch_data, _ in map_loader:
                topk_per_vector, vram = ut.get_topk_buckets_for_batch(batch_data, k, model, config.device)
                topk_per_vector.numpy()
                memory_current = ut.assign_to_buckets_vectorised(bucket_sizes, SIZE, index, chunk_size, offset, topk_per_vector, mem_tracking)
                if mem_tracking:
                    memory_usage = memory_current if memory_current > memory_usage else memory_usage
                    memory_usage = vram if memory_usage > vram else memory_usage
                offset += chunk_size
    return memory_usage

def build_full_index(bucket_sizes, SIZE, model, config: Config):
    '''
    With a trained model, go through all data (also non-sample data) and create the full index.
    The method of assigning each vector to its final bucket dependso on the chosen reassignment mode.
    '''
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
        memory_usage = map_all_to_buckets_0(index, bucket_sizes, full_data, data_batched, map_loader, SIZE, k, model, device, config.mem_tracking)

    elif reass_mode == 1: # reassign all vectors in a foward pass batch directly 
        memory_usage = map_all_to_buckets_1(map_loader, full_data, data_batched, k, index, bucket_sizes, model, device, config.mem_tracking)

    elif reass_mode == 2:
        # Do all forward passes sequentially and then do reassignments in batches
        memory_usage = map_all_to_buckets_2(bucket_sizes, SIZE, model, memory_usage, k, device, index, full_data, data_batched, chunk_size, map_loader, config.mem_tracking)

    elif reass_mode == 3:
        # Alternate fowardpasses wih batched reassignment -> vectorised assignment of a whole batch of buckets at once
        memory_usage = map_all_to_buckets_3(bucket_sizes, SIZE, model, config, memory_usage, reass_mode, k, index, full_data, data_batched, chunk_size, map_loader, config.mem_tracking)

    end = time.time()
    logging.info(f"final assignment mode:{reass_mode} took {end-start} seconds")

    return index, memory_usage

def get_all_candidate_buckets(SIZE, model, k, device, full_data, data_batched, map_loader, mem_tracking):
    '''
    For a set of data, predict the top-k candidates for each vector according to a model, and aggregate the results.
    '''
    candidate_buckets = np.zeros(shape= (SIZE, k), dtype=np.uint32) # all topk buckets per vector shape = (N, k).
    offset = 0
    for batch in full_data.get_dataset_iterator(bs = 1_000_000):
        data_batched.data = torch.from_numpy(batch).float()
        vram = ut.get_all_topk_buckets(map_loader, k, candidate_buckets, model, offset, device, mem_tracking)
        offset += len(batch)
    return candidate_buckets, vram

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
    rp_paths = []
    for i in range (config.r):
        inverted_indexes_paths.append(f"models/{config.dataset_name}_r{config.r}_k{config.k}_b{config.b}_lr{config.lr}_bs={config.batch_size}_reass={config.reass_mode}_chunk_size={config.reass_chunk_size}_e={config.epochs}_i={config.iterations}/index_model{i+1}_{config.dataset_name}_r{i+1}_k{config.k}_b{config.b}_lr{config.lr}.npy")
        offsets_paths.append(f"models/{config.dataset_name}_r{config.r}_k{config.k}_b{config.b}_lr{config.lr}_bs={config.batch_size}_reass={config.reass_mode}_chunk_size={config.reass_chunk_size}_e={config.epochs}_i={config.iterations}/offsets_model{i+1}_{config.dataset_name}_r{i+1}_k{config.k}_b{config.b}_lr{config.lr}.npy")
        model_paths.append(f"models/{config.dataset_name}_r{config.r}_k{config.k}_b{config.b}_lr{config.lr}_bs={config.batch_size}_reass={config.reass_mode}_chunk_size={config.reass_chunk_size}_e={config.epochs}_i={config.iterations}/model_{config.dataset_name}_r{i+1}_k{config.k}_b{config.b}_lr{config.lr}.pt")
        if config.query_twostep:
            rp_paths.append(f"models/{config.dataset_name}_r{config.r}_k{config.k}_b{config.b}_lr{config.lr}_bs={config.batch_size}_reass={config.reass_mode}_chunk_size={config.reass_chunk_size}_e={config.epochs}_i={config.iterations}/{config.dataset_name}_{config.datasize}_rp{config.rp_dim}_r{i+1}.npy")

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
    if config.query_twostep:
        rp_files = [np.memmap(rp_path, mode='r', shape=(SIZE, config.rp_dim), dtype=np.float32) for rp_path in rp_paths]
        index = ((indexes, offsets, rp_files), q_models)
    else:
        index = ((indexes, offsets), q_models)
    return index

def run_bliss(config: Config, mode, experiment_name, trial=None):
    '''
    Run the BLISS algorithm. Mode determines whether an index is built or whether inference is run on an existing index.
    '''

    config.experiment_name = experiment_name
    print(f"Using device: {config.device}")

    dataset = ut.get_dataset_obj(config.dataset_name, config.datasize)
    SIZE = dataset.nb
    DIM = dataset.d
    dataset.prepare()
    if mode == 'build':
        index, train_time, final_assign_time, build_time, memory_final_assignment, memory_training, load_balance, index_sizes_total, model_sizes_total, load_balances = build_index(dataset, config, trial)
        return train_time, final_assign_time, build_time, memory_final_assignment, memory_training, load_balance, index_sizes_total, model_sizes_total, load_balances
    elif mode == 'query':
        # set b if it wasn't already set in config
        b = config.b if config.b !=0 else 1024
        config.b = b
        
        logging.info("Loading models for inference")
        index = load_indexes_and_models(config, SIZE, DIM, b)
        logging.info("Reading query vectors and ground truths")
        data, test, neighbours, using_memmap = load_data_for_inference(dataset, config, SIZE, DIM)
 
        print(f"creating tensor array from Test")
        test = torch.from_numpy(test).to(torch.float32)
        
        logging.info("Starting inference")
        start = time.time()
        qstart = time.time()
        if config.query_batched and config.query_twostep:
            results, memory = query_multiple_batched_twostep(data, index, test, neighbours, config, using_memmap)
        elif config.query_batched:
            results, memory = query_multiple_batched(data, index, test, neighbours, config, using_memmap)
        elif config.query_twostep:
            results, memory = query_multiple_twostep(data, index, test, neighbours, config, using_memmap)
        else:
            results, memory = query_multiple(data, index, test, neighbours, config, using_memmap)
        print(f"querying pq={config.pq} took {time.time()-qstart}")
        end = time.time()

        total_query_time = end - start

        anns = [t[0] for t in results]
        RECALL = ut.recall(anns, neighbours)
        print(f"RECALL = {RECALL}", flush=True)
        return RECALL, results, total_query_time, memory
    