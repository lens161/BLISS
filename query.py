import math
import numpy as np
import os
import psutil # type: ignore
import time
from sklearn.random_projection import SparseRandomProjection
import torch
from faiss import IndexPQ
from multiprocessing import Pool
from torch.utils.data import DataLoader

import datasets as ds
import utils as ut
from bliss_model import BLISSDataset
from config import Config


def load_data_for_inference(dataset: ds.Dataset, config: Config, SIZE, DIM):
    '''
    For a given dataset, load the load thedataset, test data and ground truths.
    Data is loaded from an existing memmap (created during index building), so that we can return the memmap address when the dataset is too large to load into memory.
    Test data (query vectors) and ground truths (true nearest neighbours of test) are read from the original dataset files.
    '''
    using_memmap = False
    data = None
    test = dataset.get_queries()
    if SIZE <= 100_000_000:
        data = dataset.get_dataset()
        if dataset.distance() == "angular":
                data = ut.normalise_data(data)
                test = ut.normalise_data(test)
    else:
        data = dataset.get_dataset_memmap()
        using_memmap = True

    neighbours, _ = dataset.get_groundtruth()
    neighbours = neighbours[:, :config.nr_ann]
    
    return data, test, neighbours, using_memmap

def query(data, indexes, offsets, models, query_vector, neighbours, m, freq_threshold, requested_amount, process, mem_tracking, using_memmap):
    '''
    Query the index for a single vector. Get the candidate set of vectors predicted by each of the R models.
    Then, filter the candidate set based on the frequency threshold.
    If the number of candidates exceeds the requested amount, reorder using true distance computations.
    Then return the remaining set of candidates.
    '''
    predicted_buckets = np.zeros((len(models), m), dtype=np.int32)
    for i in range(len(models)):
        model = models[i]
        model.eval()
        bucket_probabilities = torch.sigmoid(model(query_vector))
        _, candidate_buckets = torch.topk(bucket_probabilities, m)
        predicted_buckets[i, :] = candidate_buckets
    unique_candidates = get_candidates_for_query_vectorised(predicted_buckets, indexes, offsets, freq_threshold)

    cand_size = len(unique_candidates)
    # TODO: fix memory tracking! no access to max_cand_size
    track_mem = False
    if mem_tracking: 
        track_mem = False if cand_size <= max_cand_size else True
        max_cand_size = max_cand_size if cand_size <= max_cand_size else cand_size

    if cand_size <= requested_amount:
        return unique_candidates, neighbours, 0, ut.recall_single(unique_candidates, neighbours), 0
    else:
        final_neighbours, dist_comps, mem = reorder(data, query_vector, np.array(unique_candidates, dtype=int), requested_amount, process, using_memmap, track_mem)
        return final_neighbours, neighbours, dist_comps, ut.recall_single(final_neighbours, neighbours), mem

def query_multiple(data, index, query_vectors, neighbours, config: Config, using_memmap):
    '''
    Run multiple queries from a set of query vectors i.e. "Test" from the ANN benchmark datsets.
    '''
    ((indexes, offsets), models) = index
    size = len(query_vectors)
    print(f"Number of query vectors: {size}")
    print(f"Number of neighbour entries: {len(neighbours)}", flush=True)
    results = [[] for i in range(len(query_vectors))]
    memory = 0
    process = psutil.Process(os.getpid())
    for i, query_vector in enumerate(query_vectors):
        print(f"\r[PID: {os.getpid()}] querying {i+1} of {size}       ", end='', flush=True)
        start = time.time()
        anns, true_nns, dist_comps, recall, current_mem = query(data, indexes, offsets, models, query_vector, neighbours[i], config.m, config.freq_threshold, config.nr_ann, process, config.mem_tracking, using_memmap)
        end = time.time()
        elapsed = end - start
        memory = current_mem if current_mem > memory else memory
        results[i] = (anns, true_nns, dist_comps, elapsed, recall)
    print("\r")
    return results, memory

def get_candidates_for_query_vectorised(predicted_buckets, model_indexes, model_offsets, freq_threshold):
    """
    For a single query:
      - predicted_buckets: NumPy array of shape (r, m) containing the m predicted buckets for each of the r models.
      - model_indexes: list or array of candidate arrays for each model (each element is a NumPy 1D array).
      - model_offsets: 2D NumPy array of shape (r, b+1) containing the bucket offsets for each model.
    Returns a 1D NumPy array of candidate IDs (with duplicates) aggregated from all r models.
    """
    r, m = predicted_buckets.shape
    model_axis = np.arange(r)[:, None]  # shape (r,1)
    start_indices = np.where(predicted_buckets == 0, 0, model_offsets[model_axis, predicted_buckets - 1])
    end_indices = model_offsets[model_axis, predicted_buckets]

    candidate_slices = [(model, start, end)
            for model in range(r)
            for start, end in zip(start_indices[model], end_indices[model])]

    lengths = [end - start for (_, start, end) in candidate_slices]
    total = sum(lengths)
    candidate_indices = np.empty(total, dtype=np.int64)
 
    pos = 0
    for (model, start, end), length in zip(candidate_slices, lengths):
        candidate_indices[pos:pos+length] = model_indexes[model][start:end]
        pos += length

    unique_vals, counts = np.unique(
        candidate_indices, return_counts=True
    )

    mask = counts >= freq_threshold
    unique_candidates = unique_vals[mask]
    return unique_candidates

def process_query_batch(data, neighbours, query_vectors, candidate_buckets, indexes, offsets, freq_threshold, requested_amount, batch_process_start, process, max_cand_size, mem_tracking, using_memmap):
    ## input: candidate_buckets np array for a batch of queries
    ## output: the ANNs for the batch of queries, dist_comps per query, recall per query
    batch_results = [[] for i in range(len(query_vectors))]
    batch_process_end = time.time()
    base_time_per_query = (batch_process_end - batch_process_start) / len(query_vectors)
    memory = 0
    for i, query in enumerate(query_vectors):
        query_start = time.time()
        # For query i, extract the predicted buckets per model (shape (r, m))
        predicted_buckets = candidate_buckets[i]
        # Use our helper to obtain the candidate set (as a 1D NumPy array)
        unique_candidates = get_candidates_for_query_vectorised(predicted_buckets, indexes, offsets, freq_threshold)

        cand_size = len(unique_candidates)
        # set memory tracking and update current largest encountered candidate set site if necessary
        track_mem = False
        if mem_tracking: 
            track_mem = False if cand_size <= max_cand_size else True
            max_cand_size = max_cand_size if cand_size <= max_cand_size else cand_size
        
        if cand_size <= requested_amount:
            query_end = time.time()
            batch_results[i] = (unique_candidates, neighbours[i], 0, (query_end-query_start) + base_time_per_query, ut.recall_single(unique_candidates, neighbours[i]))
        else:
            final_neighbours, dist_comps, current_mem = reorder(data, query, np.array(unique_candidates, dtype=int), requested_amount, process, using_memmap, track_mem)
            query_end = time.time()
            memory = current_mem if current_mem > memory else memory
            del unique_candidates
            batch_results[i] = (final_neighbours, neighbours[i], dist_comps, (query_end-query_start) + base_time_per_query, ut.recall_single(final_neighbours, neighbours[i]))
    return batch_results, memory

def query_multiple_batched(data, index, vectors, neighbours, config: Config, using_memmap):
    '''
    Run multiple queries from a set of query vectors.
    '''
    ((indexes, offsets), models) = index
    size = len(vectors)
    print(f"Number of query vectors: {size}")
    print(f"Number of neighbour entries: {len(neighbours)}", flush=True)
    nr_batches = math.ceil(size / config.query_batch_size)
    results = [[] for i in range(nr_batches)]
    # do forward passes on a batch of queries in all models and then process
    print(f"Processing queries in batches")
    
    queries_batched = BLISSDataset(vectors, device = torch.device("cpu"), mode='train')
    query_loader = DataLoader(queries_batched, batch_size=config.query_batch_size, shuffle=False, num_workers=8)
    batch_idx = 0
    memory = 0
    max_cand_size = 0
    process = psutil.Process(os.getpid())
    with torch.no_grad():
        for batch_data, batch_indices in query_loader:
            batch_process_start = time.time()
            print(f"Processing batch {batch_idx+1}/{nr_batches}")
            predicted_buckets_per_query = np.zeros((len(batch_data), len(models), config.m), dtype=np.int32)
            for i, model in enumerate(models):
                bucket_probabilities = torch.sigmoid(model(batch_data))
                _, candidate_buckets = torch.topk(bucket_probabilities, config.m, dim=1)
                predicted_buckets_per_query[:, i, :] = candidate_buckets
            batch_results, current_mem = process_query_batch(data, neighbours[batch_indices], batch_data, predicted_buckets_per_query, indexes, offsets, config.freq_threshold, config.nr_ann, batch_process_start, process, max_cand_size, config.mem_tracking, using_memmap)
            memory = current_mem if current_mem > memory else memory
            results[batch_idx] = batch_results
            batch_idx += 1

    flattened_results = [item for sublist in results for item in sublist]
    return flattened_results, memory
    
def reorder(data, query_vector, candidates, requested_amount, process, using_memmap, track_mem = False):
    '''
    Do true distance calculations on the candidate set of vectors and return the top 'requested_amount' candidates.
    ''' 
    #TODO: check if it makes sense to sort candidates before fetching from memmap to improve access pattern
    # candidates = np.sort(candidates)

    mem = 0 if not track_mem else process.memory_full_info().uss / (1024 ** 2)
    if not isinstance(data, IndexPQ):
        search_space = np.ascontiguousarray(data[candidates].copy()) if using_memmap else np.ascontiguousarray(data[candidates])
    else:
        candidates = np.asarray(candidates, dtype=np.int32)
        # search_space = np.vstack([data.reconstruct_batch(int(i)) for i in candidates])
        search_space = np.vstack(data.reconstruct_batch(candidates))
    dist_comps = len(search_space)
    query_vector = query_vector.reshape(1, -1)
    neighbours = ut.get_nearest_neighbours_in_different_dataset(search_space, query_vector, requested_amount)
    del search_space
    neighbours = neighbours[0]
    final_neighbours = candidates[neighbours]
    return final_neighbours, dist_comps, mem



###################
# TWOSTEP QUERYING
###################

def get_candidates_for_query_vectorised_twostep(predicted_buckets, model_indexes, model_offsets, model_rp_files, freq_threshold, rp_dim, timers, candidate_amount_limit):
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

def query_twostep(data, indexes, offsets, models, rp_files, query_vector, neighbours, m, expected_bucket_size, freq_threshold, requested_amount, process, mem_tracking, transformer, rp_dim, timers, using_memmap):
    '''
    Query the index for a single vector. Get the candidate set of vectors predicted by each of the R models.
    Then, filter the candidate set based on the frequency threshold.
    If the number of candidates exceeds the requested amount, reorder using true distance computations.
    Then return the remaining set of candidates.
    '''
    predicted_buckets = np.zeros((len(models), m), dtype=np.int32)
    for i in range(len(models)):
        model = models[i]
        model.eval()
        bucket_probabilities = torch.sigmoid(model(query_vector))
        _, candidate_buckets = torch.topk(bucket_probabilities, m)
        predicted_buckets[i, :] = candidate_buckets
    #TODO: figure out better scaling formula
    candidate_amount_limit = m*expected_bucket_size*4
    candidate_ids, candidate_rp_data = get_candidates_for_query_vectorised_twostep(predicted_buckets, indexes, offsets, rp_files, freq_threshold, rp_dim, timers, candidate_amount_limit)

    #TODO: track memory during rp filtering if the candidate size is large enough, depending on rp_dim
    cand_size = len(candidate_ids)
    #TODO: fix memory tracking!
    track_mem = False
    if mem_tracking: 
        track_mem = False if cand_size <= max_cand_size else True
        max_cand_size = max_cand_size if cand_size <= max_cand_size else cand_size

    if cand_size <= requested_amount:
        return candidate_ids, neighbours, 0, ut.recall_single(candidate_ids, neighbours), 0
    else:
        if candidate_rp_data is not None:
            reduced_query = apply_random_projection(query_vector, transformer)
            candidate_ids = filter_candidates(candidate_rp_data, candidate_ids, reduced_query, m, candidate_amount_limit)
            if len(candidate_ids) <= requested_amount:
                return candidate_ids, neighbours, 0, ut.recall_single(candidate_ids, neighbours), 0
        final_neighbours, dist_comps, current_mem = reorder(data, query_vector, candidate_ids, requested_amount, process, using_memmap, track_mem)
        del candidate_ids, candidate_rp_data
        return final_neighbours, neighbours, dist_comps, ut.recall_single(final_neighbours, neighbours), current_mem

def query_multiple_twostep(data, index, query_vectors, neighbours, config: Config, using_memmap):
    '''
    Run multiple queries from a set of query vectors i.e. "Test" from the ANN benchmark datsets.
    '''
    ((indexes, offsets, rp_files), models) = index
    size = len(query_vectors)
    expected_bucket_size = len(data) // config.b
    print(f"Number of query vectors: {size}")
    print(f"Number of neighbour entries: {len(neighbours)}", flush=True)
    results = [[] for i in range(len(query_vectors))]
    timers = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    memory = 0
    transformer = SparseRandomProjection(n_components=config.rp_dim)
    process = psutil.Process(os.getpid())
    for i, query_vector in enumerate(query_vectors):
        print(f"\r[PID: {os.getpid()}] querying {i+1} of {size}       ", end='', flush=True)
        start = time.time()
        anns, true_nns, dist_comps, recall, current_mem = query_twostep(data, indexes, offsets, models, rp_files, query_vector, neighbours[i], config.m, expected_bucket_size, config.freq_threshold, config.nr_ann, process, config.mem_tracking, transformer, config.rp_dim, timers, using_memmap)
        end = time.time()
        elapsed = end - start
        memory = current_mem if current_mem > memory else memory
        results[i] = (anns, true_nns, dist_comps, elapsed, recall)
    print("\r")
    
    print(f"Time spent on getting start/end indices: {timers[0]}")
    print(f"Time spent on prepping slices: {timers[1]}")
    print(f"Time spent on prepping data structures: {timers[2]}")
    print(f"Time spent on collecting candidate indices: {timers[3]}")
    print(f"Time spent on np.unique: {timers[4]}")
    print(f"Time spent on making filter mask: {timers[5]}")
    print(f"Time spent on filtering indices: {timers[6]}")
    print(f"Time spent on collecting rp vectors: {timers[7]}")
    print(f"Time spent on filtering rp vectors: {timers[8]}")

    return results, memory

def process_query_batch_twostep(data, neighbours, query_vectors, candidate_buckets, indexes, offsets, rp_files, freq_threshold, requested_amount, m, expected_bucket_size, batch_process_start, process, max_cand_size, mem_tracking, transformer, rp_dim, timers, using_memmap):
    ## input: candidate_buckets np array for a batch of queries
    ## output: the ANNs for the batch of queries, dist_comps per query, recall per query
    batch_results = [[] for i in range(len(query_vectors))]
    batch_process_end = time.time()
    base_time_per_query = (batch_process_end - batch_process_start) / len(query_vectors)
    
    #TODO: figure out better scaling formula
    candidate_amount_limit = m*expected_bucket_size*4
    memory = 0
    for i, query in enumerate(query_vectors):
        query_start = time.time()
        # For query i, extract the predicted buckets per model (shape (r, m))
        predicted_buckets = candidate_buckets[i]
        # Use our helper to obtain the candidate set (as a 1D NumPy array)
        candidate_ids, candidate_rp_data = get_candidates_for_query_vectorised_twostep(predicted_buckets, indexes, offsets, rp_files, freq_threshold, rp_dim, timers, candidate_amount_limit)

        #TODO: track memory during rp filtering if the candidate size is large enough, depending on rp_dim
        cand_size = len(candidate_ids)
        track_mem = False
        if mem_tracking: 
            track_mem = False if cand_size <= max_cand_size else True
            max_cand_size = max_cand_size if cand_size <= max_cand_size else cand_size

        if cand_size <= requested_amount:
            query_end = time.time()
            batch_results[i] = (candidate_ids, neighbours[i], 0, (query_end-query_start) + base_time_per_query), ut.recall_single(candidate_ids, neighbours[i])
        else:
            if candidate_rp_data is not None:
                reduced_query = apply_random_projection(query, transformer)
                candidate_ids = filter_candidates(candidate_rp_data, candidate_ids, reduced_query, m, candidate_amount_limit)
                if len(candidate_ids) <= requested_amount:
                    query_end = time.time()
                    batch_results[i] = (candidate_ids, neighbours[i], 0, (query_end-query_start) + base_time_per_query), ut.recall_single(candidate_ids, neighbours[i])
                    del candidate_ids, candidate_rp_data
                    return batch_results, memory
            final_neighbours, dist_comps, current_mem = reorder(data, query, candidate_ids, requested_amount, process, using_memmap, track_mem)
            memory = current_mem if current_mem > memory else memory
            del candidate_ids, candidate_rp_data
            query_end = time.time()
            batch_results[i] = (final_neighbours, neighbours[i], dist_comps, (query_end-query_start) + base_time_per_query, ut.recall_single(final_neighbours, neighbours[i]))
    return batch_results, memory

def query_multiple_batched_twostep(data, index, vectors, neighbours, config: Config, using_memmap):
    '''
    Run multiple queries from a set of query vectors i.e. "Test" from the ANN benchmark datsets.
    '''
    ((indexes, offsets, rp_files), models) = index

    size = len(vectors)
    expected_bucket_size = len(data) // config.b
    print(f"Number of query vectors: {size}")
    print(f"Number of neighbour entries: {len(neighbours)}", flush=True)
    nr_batches = math.ceil(size / config.query_batch_size)
    results = [[] for i in range(nr_batches)]
    timers = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    # do forward passes on a batch of queries in all models and then process
    print(f"Processing queries in batches")
    
    queries_batched = BLISSDataset(vectors, device = torch.device("cpu"), mode='train')
    query_loader = DataLoader(queries_batched, batch_size=config.query_batch_size, shuffle=False, num_workers=8)
    transformer = SparseRandomProjection(n_components=config.rp_dim)

    batch_idx = 0
    memory = 0
    max_cand_size = 0
    process = psutil.Process(os.getpid())
    with torch.no_grad():
        batch_process_start = time.time()
        for batch_data, batch_indices in query_loader:
            print(f"Processing batch {batch_idx+1}/{nr_batches}")
            predicted_buckets_per_query = np.zeros((len(batch_data), len(models), config.m), dtype=np.int32)
            for i, model in enumerate(models):
                bucket_probabilities = torch.sigmoid(model(batch_data))
                _, candidate_buckets = torch.topk(bucket_probabilities, config.m, dim=1)
                predicted_buckets_per_query[:, i, :] = candidate_buckets
            batch_results, current_mem = process_query_batch_twostep(data, neighbours[batch_indices], batch_data, predicted_buckets_per_query, indexes, offsets, rp_files, config.freq_threshold, config.nr_ann, config.m, expected_bucket_size, batch_process_start, process, max_cand_size, config.mem_tracking, transformer, config.rp_dim, timers, using_memmap)
            memory = current_mem if current_mem > memory else memory
            results[batch_idx] = batch_results
            batch_idx += 1

    print(f"Time spent on getting start/end indices: {timers[0]}")
    print(f"Time spent on prepping slices: {timers[1]}")
    print(f"Time spent on prepping data structures: {timers[2]}")
    print(f"Time spent on collecting candidate indices: {timers[3]}")
    print(f"Time spent on np.unique: {timers[4]}")
    print(f"Time spent on making filter mask: {timers[5]}")
    print(f"Time spent on filtering indices: {timers[6]}")
    print(f"Time spent on collecting rp vectors: {timers[7]}")
    print(f"Time spent on filtering rp vectors: {timers[8]}")
    flattened_results = [item for sublist in results for item in sublist]
    return flattened_results, memory


def apply_random_projection(query, transformer):
    query = query.reshape(1, -1)
    reduced_query = transformer.fit_transform(query)
    return reduced_query

def filter_candidates(candidate_rp_data, candidate_ids, reduced_query, m, candidate_limit):
    neighbours = ut.get_nearest_neighbours_in_different_dataset(candidate_rp_data, reduced_query, candidate_limit)
    neighbours = neighbours[0]
    filtered_candidates = candidate_ids[neighbours]
    return filtered_candidates