import math
import numpy as np
import time
import torch
from sklearn.random_projection import SparseRandomProjection
from torch.utils.data import DataLoader

import datasets as ds
import utils as ut
from bliss_model import BLISSDataset
from config import Config


def query(data, indexes, offsets, models, query_vector, neighbours, m, freq_threshold, requested_amount, using_memmap):
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

    if cand_size <= requested_amount:
        return unique_candidates, neighbours, 0, ut.recall_single(unique_candidates, neighbours), 0
    else:
        final_neighbours, dist_comps = reorder(data, query_vector, np.array(unique_candidates, dtype=int), requested_amount, using_memmap)
        return final_neighbours, neighbours, dist_comps, ut.recall_single(final_neighbours, neighbours)

def query_multiple(data, index, query_vectors, neighbours, config: Config, using_memmap):
    '''
    Run multiple queries from a set of query vectors i.e. "Test" from the ANN benchmark datsets.
    '''
    ((indexes, offsets), models) = index
    size = len(query_vectors)
    print(f"Number of query vectors: {size}")
    print(f"Number of neighbour entries: {len(neighbours)}", flush=True)
    results = [[] for i in range(len(query_vectors))]
    for i, query_vector in enumerate(query_vectors):
        print(f"\rQuerying {i+1} of {size}       ", end='', flush=True)
        start = time.time()
        anns, true_nns, dist_comps, recall = query(data, indexes, offsets, models, query_vector, neighbours[i], config.m, config.freq_threshold, config.nr_ann, using_memmap)
        end = time.time()
        elapsed = end - start
        results[i] = (anns, true_nns, dist_comps, elapsed, recall)
    print("\r")
    return results

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

def process_query_batch(data, neighbours, query_vectors, candidate_buckets, indexes, offsets, freq_threshold, requested_amount, batch_process_start, using_memmap):
    """
    Sequentially process a batch of queries on which we have already done the forward pass.
    Similar to single querying, except that we have the predicted buckets in one array for all queries in the batch.
    """
    batch_results = [[] for i in range(len(query_vectors))]
    batch_process_end = time.time()
    base_time_per_query = (batch_process_end - batch_process_start) / len(query_vectors)
    for i, query in enumerate(query_vectors):
        query_start = time.time()
        predicted_buckets = candidate_buckets[i]
        unique_candidates = get_candidates_for_query_vectorised(predicted_buckets, indexes, offsets, freq_threshold)

        cand_size = len(unique_candidates)
        
        if cand_size <= requested_amount:
            query_end = time.time()
            batch_results[i] = (unique_candidates, neighbours[i], 0, (query_end-query_start) + base_time_per_query, ut.recall_single(unique_candidates, neighbours[i]))
        else:
            final_neighbours, dist_comps = reorder(data, query, np.array(unique_candidates, dtype=int), requested_amount, using_memmap)
            query_end = time.time()
            del unique_candidates
            batch_results[i] = (final_neighbours, neighbours[i], dist_comps, (query_end-query_start) + base_time_per_query, ut.recall_single(final_neighbours, neighbours[i]))
    return batch_results

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
    print(f"Processing queries in batches")
    
    queries_batched = BLISSDataset(vectors, device = torch.device("cpu"), mode='train')
    query_loader = DataLoader(queries_batched, batch_size=config.query_batch_size, shuffle=False, num_workers=8)
    batch_idx = 0
    with torch.no_grad():
        for batch_data, batch_indices in query_loader:
            batch_process_start = time.time()
            print(f"Processing batch {batch_idx+1}/{nr_batches}")
            predicted_buckets_per_query = np.zeros((len(batch_data), len(models), config.m), dtype=np.int32)
            for i, model in enumerate(models):
                bucket_probabilities = torch.sigmoid(model(batch_data))
                _, candidate_buckets = torch.topk(bucket_probabilities, config.m, dim=1)
                predicted_buckets_per_query[:, i, :] = candidate_buckets
            batch_results = process_query_batch(data, neighbours[batch_indices], batch_data, predicted_buckets_per_query, indexes, offsets, config.freq_threshold, config.nr_ann, batch_process_start, using_memmap)
            results[batch_idx] = batch_results
            batch_idx += 1

    flattened_results = [item for sublist in results for item in sublist]
    return flattened_results
    
def reorder(data, query_vector, candidates, requested_amount, using_memmap):
    '''
    Do true distance calculations on the candidate set of vectors and return the top 'requested_amount' candidates.
    ''' 
    search_space = np.ascontiguousarray(data[candidates].copy()) if using_memmap else np.ascontiguousarray(data[candidates])
    dist_comps = len(search_space)
    query_vector = query_vector.reshape(1, -1)
    neighbours = ut.get_nearest_neighbours_in_different_dataset(search_space, query_vector, requested_amount)
    del search_space
    neighbours = neighbours[0]
    final_neighbours = candidates[neighbours]
    return final_neighbours, dist_comps


####################
# TWOSTEP QUERYING #
####################

def get_candidates_for_query_vectorised_twostep(predicted_buckets, model_indexes, model_offsets, model_rp_files, freq_threshold, rp_dim, candidate_amount_limit):
    r, m = predicted_buckets.shape
    model_axis = np.arange(r)[:, None]
    
    start_indices = np.where(predicted_buckets == 0, 0, model_offsets[model_axis, predicted_buckets - 1])
    end_indices = model_offsets[model_axis, predicted_buckets]
 
    slices = [(model, start, end)
              for model in range(r)
              for start, end in zip(start_indices[model], end_indices[model])]
 
    lengths = [end - start for (_, start, end) in slices]
    total = sum(lengths)
    candidate_indices = np.empty(total, dtype=np.int64)
    candidate_data    = None
 
    pos = 0
    for (model, start, end), length in zip(slices, lengths):
        candidate_indices[pos:pos+length] = model_indexes[model][start:end]
        pos += length

    unique_vals, first_idx, counts = np.unique(
        candidate_indices, return_index=True, return_counts=True
    )

    mask = counts >= freq_threshold
    filtered_vals = unique_vals[mask]

    if isinstance(candidate_amount_limit, int):
        if len(filtered_vals) <= candidate_amount_limit:
            return filtered_vals, None
    elif isinstance(candidate_amount_limit, tuple):
        if len(filtered_vals) <= candidate_amount_limit[0]:
            return filtered_vals, None
 
    candidate_data    = np.empty((total, rp_dim), dtype=np.float32)
    pos = 0
    for (model, start, end), length in zip(slices, lengths):
        candidate_data   [pos:pos+length] = model_rp_files[model][start:end].copy()
        pos += length

    filtered_data = candidate_data[first_idx[mask]]

    return filtered_vals, filtered_data

def query_twostep(data, indexes, offsets, models, rp_files, query_vector, neighbours, m, candidate_amount_limit, freq_threshold, requested_amount, transformer, rp_dim, using_memmap):
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
    candidate_ids, candidate_rp_data = get_candidates_for_query_vectorised_twostep(predicted_buckets, indexes, offsets, rp_files, freq_threshold, rp_dim, candidate_amount_limit)

    cand_size = len(candidate_ids)
    if cand_size <= requested_amount:
        return candidate_ids, neighbours, 0, 0, ut.recall_single(candidate_ids, neighbours), 0
    else:
        rp_dist_comps = cand_size
        if candidate_rp_data is not None:
            reduced_query = apply_random_projection(query_vector, transformer)
            candidate_ids = filter_candidates(candidate_rp_data, candidate_ids, reduced_query, candidate_amount_limit)
            if len(candidate_ids) <= requested_amount:
                del candidate_ids, candidate_rp_data
                return candidate_ids, neighbours, 0, rp_dist_comps, ut.recall_single(candidate_ids, neighbours), 0
        final_neighbours, dist_comps = reorder(data, query_vector, candidate_ids, requested_amount, using_memmap)
        del candidate_ids, candidate_rp_data
        return final_neighbours, neighbours, dist_comps, rp_dist_comps, ut.recall_single(final_neighbours, neighbours)

def query_multiple_twostep(data, index, query_vectors, neighbours, config: Config, using_memmap):
    '''
    Run multiple queries from a set of query vectors i.e. "Test" from the ANN benchmark datsets.
    '''
    ((indexes, offsets, rp_files), models) = index
    size = len(query_vectors)
    print(f"Number of query vectors: {size}")
    print(f"Number of neighbour entries: {len(neighbours)}", flush=True)
    results = [[] for i in range(len(query_vectors))]
    transformer = SparseRandomProjection(n_components=config.rp_dim, random_state=config.rp_seed)
    for i, query_vector in enumerate(query_vectors):
        print(f"\rQuerying {i+1} of {size}       ", end='', flush=True)
        start = time.time()
        anns, true_nns, dist_comps, rp_dist_comps, recall = query_twostep(data, indexes, offsets, models, rp_files, query_vector, neighbours[i], config.m, config.query_twostep_limit, config.freq_threshold, config.nr_ann, transformer, config.rp_dim, using_memmap)
        end = time.time()
        elapsed = end - start
        results[i] = (anns, true_nns, dist_comps, rp_dist_comps, elapsed, recall)
    print("\r")

    return results

def process_query_batch_twostep(data, neighbours, query_vectors, candidate_buckets, indexes, offsets, rp_files, freq_threshold, requested_amount, candidate_amount_limit, batch_process_start, transformer, rp_dim, using_memmap):
    """
    Sequentially process a batch of queries on which we have already done the forward pass, with twostep filtering enabled.
    Similar to single querying, except that we have the predicted buckets in one array for all queries in the batch.
    """
    batch_results = [[] for i in range(len(query_vectors))]
    batch_process_end = time.time()
    base_time_per_query = (batch_process_end - batch_process_start) / len(query_vectors)
    
    for i, query in enumerate(query_vectors):
        query_start = time.time()
        predicted_buckets = candidate_buckets[i]
        candidate_ids, candidate_rp_data = get_candidates_for_query_vectorised_twostep(predicted_buckets, indexes, offsets, rp_files, freq_threshold, rp_dim, candidate_amount_limit)

        cand_size = len(candidate_ids)
        if cand_size <= requested_amount:
            query_end = time.time()
            batch_results[i] = (candidate_ids, neighbours[i], 0, 0, ((query_end-query_start) + base_time_per_query), ut.recall_single(candidate_ids, neighbours[i]))
        else:
            rp_dist_comps = cand_size
            if candidate_rp_data is not None:
                reduced_query = apply_random_projection(query, transformer)
                candidate_ids = filter_candidates(candidate_rp_data, candidate_ids, reduced_query, candidate_amount_limit)
                if len(candidate_ids) <= requested_amount:
                    query_end = time.time()
                    batch_results[i] = (candidate_ids, neighbours[i], 0, rp_dist_comps, ((query_end-query_start) + base_time_per_query), ut.recall_single(candidate_ids, neighbours[i]))
                    del candidate_ids, candidate_rp_data
                    continue
            final_neighbours, dist_comps = reorder(data, query, candidate_ids, requested_amount, using_memmap)
            del candidate_ids, candidate_rp_data
            query_end = time.time()
            batch_results[i] = (final_neighbours, neighbours[i], dist_comps, rp_dist_comps, ((query_end-query_start) + base_time_per_query), ut.recall_single(final_neighbours, neighbours[i]))
    return batch_results

def query_multiple_batched_twostep(data, index, vectors, neighbours, config: Config, using_memmap):
    '''
    Run multiple queries from a set of query vectors i.e. "Test" from the ANN benchmark datsets.
    '''
    ((indexes, offsets, rp_files), models) = index

    size = len(vectors)
    print(f"Number of query vectors: {size}")
    print(f"Number of neighbour entries: {len(neighbours)}", flush=True)
    nr_batches = math.ceil(size / config.query_batch_size)
    results = [[] for i in range(nr_batches)]

    print(f"Processing queries in batches")
    
    queries_batched = BLISSDataset(vectors, device = torch.device("cpu"), mode='train')
    query_loader = DataLoader(queries_batched, batch_size=config.query_batch_size, shuffle=False, num_workers=8)
    transformer = SparseRandomProjection(n_components=config.rp_dim, random_state=config.rp_seed)

    batch_idx = 0
    with torch.no_grad():
        batch_process_start = time.time()
        for batch_data, batch_indices in query_loader:
            print(f"Processing batch {batch_idx+1}/{nr_batches}")
            predicted_buckets_per_query = np.zeros((len(batch_data), len(models), config.m), dtype=np.int32)
            for i, model in enumerate(models):
                bucket_probabilities = torch.sigmoid(model(batch_data))
                _, candidate_buckets = torch.topk(bucket_probabilities, config.m, dim=1)
                predicted_buckets_per_query[:, i, :] = candidate_buckets
            batch_results = process_query_batch_twostep(data, neighbours[batch_indices], batch_data, predicted_buckets_per_query, indexes, offsets, rp_files, config.freq_threshold, config.nr_ann, config.query_twostep_limit, batch_process_start, transformer, config.rp_dim, using_memmap)
            results[batch_idx] = batch_results
            batch_idx += 1

    flattened_results = [item for sublist in results for item in sublist]
    return flattened_results


def apply_random_projection(query, transformer):
    query = query.reshape(1, -1)
    reduced_query = transformer.fit_transform(query)
    return reduced_query

def filter_candidates(candidate_rp_data, candidate_ids, reduced_query, candidate_limit):
    if isinstance(candidate_limit, float):
        candidate_limit = round(candidate_limit*len(candidate_ids))
    elif isinstance(candidate_limit, tuple):
        candidate_limit = max(round(candidate_limit[1]*len(candidate_ids)), candidate_limit[0])
    
    neighbours = ut.get_nearest_neighbours_in_different_dataset(candidate_rp_data, reduced_query, candidate_limit)
    neighbours = neighbours[0]
    filtered_candidates = candidate_ids[neighbours]
    return filtered_candidates
