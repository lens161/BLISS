import math
import numpy as np
import os
import psutil # type: ignore
import sys
import time
import torch
from collections import Counter
from faiss import IndexPQ
from functools import partial
from multiprocessing import Pool
from pympler import asizeof
from sklearn.random_projection import SparseRandomProjection
from torch.utils.data import DataLoader

import datasets as ds
from bliss_model import BLISSDataset
import utils as ut
from config import Config

def load_data_for_inference(dataset: ds.Dataset, config: Config, SIZE, DIM):
    '''
    For a given dataset, load the load thedataset, test data and ground truths.
    Data is loaded from an existing memmap (created during index building), so that we can return the memmap address when the dataset is too large to load into memory.
    Test data (query vectors) and ground truths (true nearest neighbours of test) are read from the original dataset files.
    '''
    using_memmap = False
    data = None
    if SIZE <= 10_000_000:
        data = dataset.get_dataset() 
    else:
        data = dataset.get_dataset_memmap()
        using_memmap = True

    test = dataset.get_queries()
    neighbours, _ = dataset.get_groundtruth()
    neighbours = neighbours[:, :config.nr_ann]

    if dataset.distance() == "angular":
        test = ut.normalise_data(test)
    
    return data, test, neighbours, using_memmap

def query(data, index, query_vector, neighbours, m, freq_threshold, requested_amount):
    '''
    Query the index for a single vector. Get the candidate set of vectors predicted by each of the R models.
    Then, filter the candidate set based on the frequency threshold.
    If the number of candidates exceeds the requested amount, reorder using true distance computations.
    Then return the remaining set of candidates.
    '''
    ((indexes, offsets), models) = index
    #
    forward_pass_start = time.time()
    #
    predicted_buckets = np.zeros((len(models), m), dtype=np.int32)
    for i in range(len(models)):
        model = models[i]
        model.eval()
        bucket_probabilities = torch.sigmoid(model(query_vector))
        _, candidate_buckets = torch.topk(bucket_probabilities, m)
        predicted_buckets[i, :] = candidate_buckets
    #
    forward_pass_end = time.time()
    forward_pass_time = forward_pass_end - forward_pass_start
    collecting_candidates_start = time.time()
    #
    candidates = get_candidates_for_query_vectorised(predicted_buckets, indexes, offsets)
    #
    collecting_candidates_end = time.time()
    collecting_candidates_time = collecting_candidates_end - collecting_candidates_start
    #
    counts = np.bincount(candidates)
    unique_candidates = np.where(counts >= freq_threshold)[0]

    # final_results = [key for key, value in candidates.items() if value >= freq_threshold]
    # print (f"final results = {len(final_results)}")
    if len(unique_candidates) <= requested_amount:
        # TODO: remove additional return values when removing timers
        return unique_candidates, neighbours, 0, ut.recall_single(unique_candidates, neighbours), forward_pass_time, collecting_candidates_time, 0, 0, 0
    else:
        #
        reorder_s = time.time()
        #
        # TODO: also implement memory measurement here if we keep it
        final_neighbours, dist_comps, true_nns_time, fetch_data_time, mem = reorder(data, query_vector, np.array(unique_candidates, dtype=int), requested_amount)
        #
        reorder_e = time.time()
        reordering_time = (reorder_e - reorder_s)
        #
        # TODO: remove additional return values when removing timers
        return final_neighbours, neighbours, dist_comps, ut.recall_single(final_neighbours, neighbours), forward_pass_time, collecting_candidates_time, reordering_time, true_nns_time, fetch_data_time

def query_multiple(data, index, vectors, neighbours, config: Config):
    '''
    Run multiple queries from a set of query vectors i.e. "Test" from the ANN benchmark datsets.
    '''
    size = len(vectors)
    print(f"Number of query vectors: {size}")
    print(f"Number of neighbour entries: {len(neighbours)}", flush=True)
    results = [[] for i in range(len(vectors))]
    #
    forward_pass_sum = 0
    collecting_candidates_sum = 0
    reordering_sum = 0
    true_nns_sum = 0
    fetch_data_sum = 0
    #
    for i, vector in enumerate(vectors):
        sys.stdout.write(f"\r[PID: {os.getpid()}] querying {i+1} of {size}       ")
        sys.stdout.flush()
        start = time.time()
        anns, true_nns, dist_comps, recall, forward_pass_time, collecting_candidates_time, reordering_time, true_nns_time, fetch_data_time = query(data, index, vector, neighbours[i], config.m, config.freq_threshold, config.nr_ann)
        end = time.time()
        #
        forward_pass_sum += forward_pass_time
        collecting_candidates_sum += collecting_candidates_time
        reordering_sum += reordering_time
        true_nns_sum += true_nns_time
        fetch_data_sum += fetch_data_time
        #
        elapsed = end - start
        results[i] = (anns, true_nns, dist_comps, elapsed, recall)
    print("\r")
    #
    print(f"Time spent on forward passes: {forward_pass_sum}")
    print(f"Time spent collecting candidates: {collecting_candidates_sum}")
    print(f"Time spent on true nns: {true_nns_sum}")
    print(f"Time spent on fetching data: {fetch_data_sum}")
    print(f"Time spent on other reordering crap: {reordering_sum - true_nns_sum - fetch_data_sum}")
    #
    return results

def get_candidates_for_query_vectorised(predicted_buckets, model_indexes, model_offsets, model_rp_files, freq_threshold, rp_dim, timers, candidate_amount_limit):
    r, m = predicted_buckets.shape
    model_axis = np.arange(r)[:, None]
    
    s = time.time()
    start_indices = np.where(predicted_buckets == 0, 0, model_offsets[model_axis, predicted_buckets - 1])
    end_indices = model_offsets[model_axis, predicted_buckets]
    e = time.time()
    timers[0]+=(e-s)
 
    s = time.time()
    slices = [(model, start, end)
              for model in range(r)
              for start, end in zip(start_indices[model], end_indices[model])]
    e = time.time()
    timers[1]+=(e-s)
 
    s = time.time()
    lengths = [end - start for (_, start, end) in slices]
    total = sum(lengths)
    candidate_indices = np.empty(total, dtype=np.int64)
    candidate_data    = None
    e = time.time()
    timers[2]+=(e-s)
 
    s = time.time()
    pos = 0
    for (model, start, end), length in zip(slices, lengths):
        candidate_indices[pos:pos+length] = model_indexes[model][start:end]
        pos += length
    e = time.time()
    timers[3]+=(e-s)

    s = time.time()
    unique_vals, first_idx, counts = np.unique(
        candidate_indices, return_index=True, return_counts=True
    )
    e = time.time()
    timers[4] += (e-s)

    s = time.time()
    mask = counts >= freq_threshold
    e = time.time()
    timers[5] += (e-s)

    s = time.time()
    filtered_vals = unique_vals[mask]
    e = time.time()
    timers[6] += (e-s)

    if len(filtered_vals) <= candidate_amount_limit:
        return filtered_vals, None
 
    s = time.time()
    candidate_data    = np.empty((total, rp_dim), dtype=np.float32)
    pos = 0
    for (model, start, end), length in zip(slices, lengths):
        candidate_data   [pos:pos+length] = model_rp_files[model][start:end].copy()
        pos += length
    e = time.time()
    timers[7] += (e-s)

    s = time.time()
    filtered_data = candidate_data[first_idx[mask]]
    e = time.time()
    timers[8]+=(e-s)

    return filtered_vals, filtered_data

def process_query_batch_twostep(data, neighbours, query_vectors, candidate_buckets, indexes, offsets, rp_files, freq_threshold, requested_amount, m, expected_bucket_size, batch_process_start, transformer, rp_dim, timers, using_memmap):
    ## input: candidate_buckets np array for a batch of queries
    ## output: the ANNs for the batch of queries, dist_comps per query, recall per query
    batch_results = [[] for i in range(len(query_vectors))]
    batch_process_end = time.time()
    base_time_per_query = (batch_process_end - batch_process_start) / len(query_vectors)
    candidate_amount_limit = m*expected_bucket_size*2
    getting_candidates_time = 0
    reordering_time = 0
    true_nns_sum = 0
    fetch_data_sum = 0
    memory = 0
    for i, query in enumerate(query_vectors):
        query_start = time.time()
        # For query i, extract the predicted buckets per model (shape (r, m))
        predicted_buckets = candidate_buckets[i]
        # Use our helper to obtain the candidate set (as a 1D NumPy array)
        getting_candidates_start = time.time()
        candidate_ids, candidate_rp_data = get_candidates_for_query_vectorised(predicted_buckets, indexes, offsets, rp_files, freq_threshold, rp_dim, timers, candidate_amount_limit)
        getting_candidates_end = time.time()
        getting_candidates_time += (getting_candidates_end - getting_candidates_start)

        if len(candidate_ids) <= requested_amount:
            query_end = time.time()
            batch_results[i] = (candidate_ids, neighbours[i], 0, (query_end-query_start) + base_time_per_query), ut.recall_single(candidate_ids, neighbours[i])
        else:
            if candidate_rp_data is not None:
                reduced_query = apply_random_projection(query, transformer)
                candidate_ids = filter_candidates(candidate_rp_data, candidate_ids, reduced_query, m, expected_bucket_size)
                if len(candidate_ids) <= requested_amount:
                    query_end = time.time()
                    batch_results[i] = (candidate_ids, neighbours[i], 0, (query_end-query_start) + base_time_per_query), ut.recall_single(candidate_ids, neighbours[i])
                    del candidate_ids, candidate_rp_data
                    return batch_results, getting_candidates_time, true_nns_sum, reordering_time, fetch_data_sum, memory
            reordering_start = time.time()
            final_neighbours, dist_comps, true_nns_time, fetch_data_time, current_mem = reorder(data, query, candidate_ids, requested_amount, using_memmap)
            memory = current_mem if current_mem > memory else memory
            del candidate_ids, candidate_rp_data
            query_end = time.time()
            batch_results[i] = (final_neighbours, neighbours[i], dist_comps, (query_end-query_start) + base_time_per_query, ut.recall_single(final_neighbours, neighbours[i]))
            reordering_time += (query_end - reordering_start)
            true_nns_sum += true_nns_time
            fetch_data_sum += fetch_data_time
    return batch_results, getting_candidates_time, true_nns_sum, reordering_time, fetch_data_sum, memory

def query_multiple_batched(data, index, vectors, neighbours, config: Config, using_memmap):
    '''
    Run multiple queries from a set of query vectors i.e. "Test" from the ANN benchmark datsets.
    '''
    if config.query_twostep:
        ((indexes, offsets, rp_files), models) = index
    else:
        ((indexes, offsets), models) = index
    all_bucket_sizes = np.zeros(shape = (config.r, config.b), dtype=np.uint32)
    for r, offset in enumerate(offsets):
        all_bucket_sizes[r] = np.diff(offset, prepend=0)

    size = len(vectors)
    expected_bucket_size = len(data) // config.b
    print(f"Number of query vectors: {size}")
    print(f"Number of neighbour entries: {len(neighbours)}", flush=True)
    nr_batches = math.ceil(size / config.query_batch_size)
    results = [[] for i in range(nr_batches)]
    #
    forward_pass_sum = 0
    collecting_candidates_sum = 0
    reordering_sum = 0
    true_nns_sum = 0
    fetch_data_sum = 0
    timers = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    #

    # do forward passes on a batch of queries in all models and then process
    print(f"Processing queries in batches")
    
    queries_batched = BLISSDataset(vectors, device = torch.device("cpu"), mode='train')
    query_loader = DataLoader(queries_batched, batch_size=config.query_batch_size, shuffle=False, num_workers=8)
    transformer = SparseRandomProjection(n_components=config.rp_dim)

    batch_idx = 0
    memory = 0
    with torch.no_grad():
        batch_process_start = time.time()
        for batch_data, batch_indices in query_loader:
            print(f"Processing one query batch")
            forward_pass_start = time.time()
            predicted_buckets_per_query = np.zeros((len(batch_data), len(models), config.m), dtype=np.int32)
            for i, model in enumerate(models):
                bucket_probabilities = torch.sigmoid(model(batch_data))
                _, candidate_buckets = torch.topk(bucket_probabilities, config.m, dim=1)
                predicted_buckets_per_query[:, i, :] = candidate_buckets
            forward_pass_end = time.time()
            forward_pass_sum += (forward_pass_end - forward_pass_start)
            batch_results, collecting_candidates_time, true_nns_time, reordering_time, fetch_data_time, current_mem = process_query_batch_twostep(data, neighbours[batch_indices], batch_data, predicted_buckets_per_query, indexes, offsets, rp_files, config.freq_threshold, config.nr_ann, config.m, expected_bucket_size, batch_process_start, transformer, config.rp_dim, timers, using_memmap)
            memory = current_mem if current_mem > memory else memory
            collecting_candidates_sum += collecting_candidates_time
            true_nns_sum += true_nns_time
            reordering_sum += reordering_time
            fetch_data_sum += fetch_data_time
            results[batch_idx] = batch_results
            batch_idx += 1

    print(f"Time spent on forward passes: {forward_pass_sum}")
    print(f"Time spent collecting candidates: {collecting_candidates_sum}")
    print(f"Time spent on true nns: {true_nns_sum}")
    print(f"Time spent on fetching candidates: {fetch_data_sum}")
    print(f"Time spent on other reordering crap: {reordering_sum - true_nns_sum - fetch_data_sum}")

    print(f"Time spent on getting start/end indices: {timers[0]}")
    print(f"Time spent on prepping slices: {timers[1]}")
    print(f"Time spent on prepping data structures: {timers[2]}")
    print(f"Time spent on collecting candidate indices: {timers[3]}")
    # print(f"Time spent on bincount and filtering candidate indices: {timers[4]}")
    print(f"Time spent on np.unique: {timers[4]}")
    print(f"Time spent on making filter mask: {timers[5]}")
    print(f"Time spent on filtering indices: {timers[6]}")
    print(f"Time spent on collecting rp vectors: {timers[7]}")
    print(f"Time spent on filtering rp vectors: {timers[8]}")
    flattened_results = [item for sublist in results for item in sublist]
    return flattened_results, memory
    
def reorder(data, query_vector, filtered_candidates, requested_amount, using_memmap):
    '''
    Do true distance calculations on the candidate set of vectors and return the top 'requested_amount' candidates.
    ''' 
    #TODO: check if it makes sense to sort candidates before fetching from memmap to improve access pattern
    # candidates = np.sort(candidates)

    #TODO: figure out new way to take these measurements as they cause slowdown
    # process = psutil.Process(os.getpid())
    # mem = process.memory_full_info().uss / (1024 ** 2)
    mem = 0
    fetch_data_s = time.time()
    search_space = np.ascontiguousarray(data[filtered_candidates].copy()) if using_memmap else np.ascontiguousarray(data[filtered_candidates])
    fetch_data_e = time.time()
    dist_comps = len(search_space)
    query_vector = query_vector.reshape(1, -1)
    true_nns_s = time.time()
    neighbours = ut.get_nearest_neighbours_in_different_dataset(search_space, query_vector, requested_amount)
    true_nns_e = time.time()
    neighbours = neighbours[0]
    final_neighbours = filtered_candidates[neighbours]
    del search_space
    return final_neighbours, dist_comps, (true_nns_e-true_nns_s), (fetch_data_e-fetch_data_s), mem

def apply_random_projection(query, transformer):
    query = query.reshape(1, -1)
    reduced_query = transformer.fit_transform(query)
    return reduced_query

def filter_candidates(candidate_rp_data, candidate_ids, reduced_query, m, expected_bucket_size):
    neighbours = ut.get_nearest_neighbours_in_different_dataset(candidate_rp_data, reduced_query, m*expected_bucket_size*2)
    neighbours = neighbours[0]
    filtered_candidates = candidate_ids[neighbours]
    return filtered_candidates