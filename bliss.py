import numpy as np
from config import Config
import time
import torch
from torch.utils.data import DataLoader
from sklearn.utils import murmurhash3_32 as mmh3
from utils import *
import psutil  # type: ignore
import os
import logging
from query import query_multiple, query_multiple_parallel, recall
from bliss_model import BLISS_NN, BLISSDataset
from train import make_ground_truth_labels, train_model
import resource
import sys
from pympler import asizeof
    
def build_index(dataset: Dataset, config: Config):
        
    print("bulding index...")
    logging.info(f"Started building index")
    SIZE = dataset.nb
    DIM = dataset.d

    sample_size = SIZE if SIZE < 2_000_000 else 1_000_000

    memmap_path = f"memmaps/{config.dataset_name}_{config.datasize}.npy"
    mmp_shape = (SIZE, DIM)
    print(f"mmp shape = {mmp_shape}")
    mmp = None
    if not os.path.exists(memmap_path):
        mmp = np.memmap(memmap_path, mode ="w+", shape=mmp_shape, dtype=np.float32)
        index = 0
        if SIZE >= 10_000_000:
            for batch in dataset.get_dataset_iterator(1_000_000):
                process = psutil.Process(os.getpid())
                memory_usage = process.memory_info().rss / (1024 ** 2)
                log_mem(f"while loading for memmap batch: ", memory_usage, config.memlog_path)
                print(f"mem usage while loading batch: {memory_usage}")
                batch_size = len(batch)
                mmp[index: index + batch_size] = batch
                del batch
                index += batch_size
        else:
            data = dataset.get_dataset()[:]
            if dataset.distance() == "angular":
                norms = np.linalg.norm(data, axis=1, keepdims=True)
                data= data / norms
            mmp[:] = data
        mmp.flush()
    
    sample = np.zeros(shape=(sample_size, DIM))
    mmp = np.memmap(memmap_path, mode = 'r', shape = mmp_shape, dtype=np.float32)
    if sample_size!=SIZE:
        random_order = np.arange(SIZE)
        np.random.seed(42)
        np.random.shuffle(random_order)
        sample_indexes = np.sort(random_order[:sample_size])
        sample[:] = mmp[sample_indexes, :]
    else:
        sample[:] = mmp
    print(f"sample size = {len(sample)}")

    print("finding neighbours...", flush=True)
    logging.info(f"Finding ground truths for train vectors")
    neighbours = get_train_nearest_neighbours_from_file(sample, config.nr_train_neighbours, sample_size, config.dataset_name)

    labels = []
    dataset = BLISSDataset(sample, labels, config.device)

    final_index = []
    time_per_r = [] 
    process = psutil.Process(os.getpid())
    memory_usage = process.memory_info().rss / (1024 ** 2)
    bucket_size_stats = []
    log_mem(f"before_building_global={config.global_reass}_shuffle={config.shuffle}", memory_usage, config.memlog_path)
    for r in range(config.r):
        logging.info(f"Training model {r+1}")
        start = time.time()

        sample_buckets, bucket_sizes = assign_initial_buckets(sample_size, r, config.b)
        logging.info(f"Random starting buckets initialized")
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
        train_model(model, dataset, sample_buckets, sample_size, bucket_sizes, neighbours, r, SIZE, config)
        model_path = save_model(model, config.dataset_name, r+1, config.r, config.k, config.b, config.lr, config.shuffle, config.global_reass)
        print(f"model {r+1} saved to {model_path}.")

        model.eval()
        model.to("cpu")
        index = None
        if sample_size < SIZE:
            # # OLD VERSION (not batched)
            # if config.datasize == 1000:
            #     for batch in dataset.get_dataset_iterator():
            #         map_all_to_buckets_unbatched(batch, config.k, index, bucket_sizes, model)
            # else:
                # map_all_to_buckets_unbatched(train, config.k, index, bucket_sizes, model)

            # NEW VERSION (batched) TO-DO: idx resets to 0 with every new bath -> fix this 
            bucket_sizes[:] = 0 
            index = np.zeros(SIZE, dtype=int)
            full_data = get_dataset_obj(config.dataset_name, config.datasize)
            data_batched = BLISSDataset(None, None, device = torch.device("cpu"), mode='map')
            map_loader = DataLoader(data_batched, batch_size=config.batch_size, shuffle=False, num_workers=8)
            global_idx = 0
            for batch in full_data.get_dataset_iterator(bs=1_000_000):
                data_batched.data = batch
                map_all_to_buckets(map_loader, config.k, index, bucket_sizes, model, global_idx)
        else:
            index = sample_buckets
        offsets = np.cumsum(bucket_sizes)

        inverted_index = np.zeros(SIZE, dtype=np.uint32)
        inverted_index = np.argsort(index)
        np.set_printoptions(threshold=np.inf, suppress=True)
        print(f"bucket_sizes sum = {np.sum(bucket_sizes)}", flush=True)
        print(bucket_sizes)
        # inverted_index = invert_index(index, config.b) 
        index_path = save_inverted_index(inverted_index, offsets, config.dataset_name, r+1, config.r, config.k, config.b, config.lr, config.shuffle, config.global_reass)
        final_index.append((index_path, model_path))
        end = time.time()
        time_per_r.append(end - start)
        bucket_size_stats.append(bucket_sizes)

    build_time = time.time() - start
    process = psutil.Process(os.getpid())
    # memory_usage = process.memory_info().rss / (1024 ** 2)
    peak_mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    load_balance = calc_load_balance(bucket_size_stats)
    log_mem(f"after_building_global={config.global_reass}_shuffle={config.shuffle}", memory_usage, config.memlog_path)
    return final_index, time_per_r, build_time, peak_mem, load_balance

def assign_initial_buckets(train_size, r, B):
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
    
    return np.array(index, dtype=np.uint32), bucket_sizes

def map_all_to_buckets_unbatched(data, k, index, bucket_sizes, map_model):
    data = torch.from_numpy(data).float()
    data = data.to("cpu")

    for i, vector in enumerate(data):
        scores = map_model(vector)
        probabilities = torch.sigmoid(scores)
        reassign_vector_to_bucket(probabilities, index, bucket_sizes, k, i)

def map_all_to_buckets(map_loader, k, index, bucket_sizes, map_model, offset):
    logging.info(f"Mapping all train vectors to buckets")
    with torch.no_grad():
        for batch, batch_indices in map_loader:
            scores = map_model(batch)
            probabilities = torch.sigmoid(scores)

            for probability_vector, idx in zip(probabilities, batch_indices):
                global_idx = idx + offset
                if offset % 1000000 == 0:
                    print(f"no worries i am still alive, reasigning vector {idx}", flush=True)
                reassign_vector_to_bucket(probability_vector, index, bucket_sizes, k, global_idx)

def invert_index(index, B):
    inverted_index = [[] for _ in range(B)]
    for i, bucket in enumerate(index):
        inverted_index[bucket].append(i)
    return inverted_index

def run_bliss(config: Config, mode, experiment_name):

    config.experiment_name = experiment_name
    dataset_name = config.dataset_name
    print(f"Using device: {config.device}")
    print(dataset_name)

    if not os.path.exists(f"results/{experiment_name}/"):
        os.mkdir(f"results/{experiment_name}/")
    MEMLOG_PATH = f"results/{experiment_name}/{experiment_name}_memory_log.csv"
    config.memlog_path = MEMLOG_PATH

    inverted_indexes_paths = []
    dataset = get_dataset_obj(dataset_name, config.datasize)
    SIZE = dataset.nb
    DIM = dataset.d
    dataset.prepare()
    if mode == 'build':
        index, time_per_r, build_time, memory_usage, load_balance = build_index(dataset, config)
        usage = resource.getrusage(resource.RUSAGE_SELF)
        peak_mem = usage.ru_maxrss / 1_000_000 if sys.platform == 'darwin' else usage.ru_maxrss / 1000
        log_mem("peak_mem_building", peak_mem, MEMLOG_PATH)
        inverted_indexes_paths, model_paths = zip(*index)
        return time_per_r, build_time, memory_usage, load_balance
    elif mode == 'query':
        logging.info("Loading models for inference")
        b = config.b if config.b !=0 else 1024
        config.b = b
        inverted_indexes_paths = []
        offsets_paths = []
        model_paths = []
        for i in range (config.r):
            inverted_indexes_paths.append(f"models/{dataset_name}_r{config.r}_k{config.k}_b{config.b}_lr{config.lr}_shf={config.shuffle}_gr={config.global_reass}/index_model{i+1}_{dataset_name}_r{i+1}_k{config.k}_b{config.b}_lr{config.lr}.npy")
            offsets_paths.append(f"models/{dataset_name}_r{config.r}_k{config.k}_b{config.b}_lr{config.lr}_shf={config.shuffle}_gr={config.global_reass}/offsets_model{i+1}_{dataset_name}_r{i+1}_k{config.k}_b{config.b}_lr{config.lr}.npy")
            model_paths.append(f"models/{dataset_name}_r{config.r}_k{config.k}_b{config.b}_lr{config.lr}_shf={config.shuffle}_gr={config.global_reass}/model_{dataset_name}_r{i+1}_k{config.k}_b{config.b}_lr{config.lr}.pt")

        indexes = np.zeros(shape = (config.r, SIZE), dtype=np.uint32)
        offsets = np.zeros(shape = (config.r, config.b), dtype=np.uint32)
        for i, (inv_path, off_path) in enumerate(zip(inverted_indexes_paths, offsets_paths)):
            inds_load = np.load(inv_path)
            ind = np.array(inds_load)
            indexes[i] = ind
            offs_load = np.load(off_path)
            off = np.array(offs_load)
            offsets[i] = off
        
        memmap_path = f"memmaps/{dataset_name}_{config.datasize}.npy"
        data = np.memmap(memmap_path, mode='r', shape=(SIZE, DIM), dtype=np.float32) if SIZE >10_000_000 else np.memmap(memmap_path,shape=(SIZE, DIM), mode='r', dtype=np.float32)[:]

        q_models = [load_model(model_path, DIM, b) for model_path in model_paths]
        index = ((indexes, offsets), q_models)

        logging.info("Reading query vectors and ground truths")
        test = dataset.get_queries()
        neighbours, _ = dataset.get_groundtruth()

        if dataset.distance() == "angular":
            norms = np.linalg.norm(test, axis=1, keepdims=True)
            test = test / norms

        print(f"creating tensor array from Test")
        test = torch.from_numpy(test).float()

        logging.info("Starting inference")
        num_workers = 8
        start = time.time()
        # results = query_multiple(data, index, test, neighbours, config.m, config.freq_threshold, config.nr_train_neighbours)
        results = query_multiple_parallel(data, index, test, neighbours, config.m, config.freq_threshold, config.nr_train_neighbours, num_workers)
        end = time.time()

        total_query_time = end - start

        anns = [t[0] for t in results]
        RECALL = recall(anns, neighbours)
        print(f"RECALL = {RECALL}", flush=True)
        usage = resource.getrusage(resource.RUSAGE_SELF)
        peak_mem = usage.ru_maxrss / 1_000_000 if sys.platform == 'darwin' else usage.ru_maxrss / 1000
        log_mem("peak_mem_querying", peak_mem, MEMLOG_PATH)
        return RECALL, results, total_query_time