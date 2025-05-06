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
from torch.utils.data import DataLoader

from bliss_model import BLISSDataset
import utils as ut
from config import Config


def load_data_for_inference(dataset, config: Config, SIZE, DIM):
    '''
    For a given dataset, load the load thedataset, test data and ground truths.
    Data is loaded from an existing memmap (created during index building), so that we can return the memmap address when the dataset is too large to load into memory.
    Test data (query vectors) and ground truths (true nearest neighbours of test) are read from the original dataset files.
    '''
    # keep data as a memmap if the dataset is too large, otherwise load into memory fully
    memmap_path = f"memmaps/{config.dataset_name}_{config.datasize}.npy"
    data = np.memmap(memmap_path, mode='r', shape=(SIZE, DIM), dtype=np.float32) if SIZE > 10_000_000 else np.ascontiguousarray(np.memmap(memmap_path,shape=(SIZE, DIM), mode='r', dtype=np.float32).copy())

    test = dataset.get_queries()
    neighbours, _ = dataset.get_groundtruth()
    neighbours = neighbours[:, :config.nr_ann]

    if dataset.distance() == "angular":
        test = ut.normalise_data(test)
    
    return data, test, neighbours

def query(data, index, query_vector, neighbours, m, freq_threshold, requested_amount):
    '''
    Query the index for a single vector. Get the candidate set of vectors predicted by each of the R models.
    Then, filter the candidate set based on the frequency threshold.
    If the number of candidates exceeds the requested amount, reorder using true distance computations.
    Then return the remaining set of candidates.
    '''
    ((indexes, offsets), models) = index
    predicted_buckets = np.zeros((len(models), m), dtype=np.int32)
    for i in range(len(models)):
        model = models[i]
        model.eval()
        bucket_probabilities = torch.sigmoid(model(query_vector))
        _, candidate_buckets = torch.topk(bucket_probabilities, m)
        predicted_buckets[i, :] = candidate_buckets
    candidates = get_candidates_for_query_vectorised(predicted_buckets, indexes, offsets)
    counts = np.bincount(candidates)
    unique_candidates = np.where(counts >= freq_threshold)[0]

    # final_results = [key for key, value in candidates.items() if value >= freq_threshold]
    # print (f"final results = {len(final_results)}")
    if len(unique_candidates) <= requested_amount:
        # TODO: remove additional return values when removing timers
        return unique_candidates, neighbours, 0, ut.recall_single(unique_candidates, neighbours), 0, 0, 0
    else:
        # TODO: also implement memory measurement here if we keep it
        final_neighbours, dist_comps, mem = reorder(data, query_vector, np.array(unique_candidates, dtype=int), requested_amount)
        # TODO: remove additional return values when removing timers
        return final_neighbours, neighbours, dist_comps, ut.recall_single(final_neighbours, neighbours), mem

def query_multiple(data, index, vectors, neighbours, config: Config):
    '''
    Run multiple queries from a set of query vectors i.e. "Test" from the ANN benchmark datsets.
    '''
    size = len(vectors)
    print(f"Number of query vectors: {size}")
    print(f"Number of neighbour entries: {len(neighbours)}", flush=True)
    results = [[] for i in range(len(vectors))]
    for i, vector in enumerate(vectors):
        sys.stdout.write(f"\r[PID: {os.getpid()}] querying {i+1} of {size}       ")
        sys.stdout.flush()
        start = time.time()
        anns, true_nns, dist_comps, recall, memory = query(data, index, vector, neighbours[i], config.m, config.freq_threshold, config.nr_ann)
        end = time.time()
        elapsed = end - start
        results[i] = (anns, true_nns, dist_comps, elapsed, recall)
    print("\r")
    return results, memory

def get_candidates_for_query_vectorised(predicted_buckets, model_indexes, model_offsets):
    """
    For a single query:
      - predicted_buckets: NumPy array of shape (r, m) containing the m predicted buckets for each of the r models.
      - model_indexes: list or array of candidate arrays for each model (each element is a NumPy 1D array).
      - model_offsets: 2D NumPy array of shape (r, b+1) containing the bucket offsets for each model.
    Returns a 1D NumPy array of candidate IDs (with duplicates) aggregated from all r models.
    """
    # Number of models (r) and number of bucket predictions per model (m)
    r, m = predicted_buckets.shape
 
    # Create an array for model indices so that we can broadcast over the (r, m) predictions.
    model_axis = np.arange(r)[:, None]  # shape (r,1)
 
    # For each (model, predicted bucket) we want to compute a start index:
    # If bucket == 0, start index is 0, else it is offsets[model, bucket-1]
    start_indices = np.where(
        predicted_buckets == 0,
        0,
        model_offsets[model_axis, predicted_buckets - 1]
    )
 
    # Similarly, get the end indices for each bucket:
    end_indices = model_offsets[model_axis, predicted_buckets]
    # At this point, 'start_indices' and 'end_indices' are arrays of shape (r, m)
    # Their difference gives the number of candidates in each predicted bucket.
    # Although these differences are computed vectorized, the slices themselves will be ragged
    # (i.e. different lengths per bucket) so we then extract each slice.
    candidate_slices = [
        model_indexes[model][start: end]
        for model in range(r)
        for start, end in zip(start_indices[model], end_indices[model])
    ]
    # Concatenate all the candidate slices (this gives duplicates if same candidate appears in multiple buckets).
    if candidate_slices:
        candidates = np.concatenate(candidate_slices)
    else:
        candidates = np.empty(0, dtype=np.int32)
    return candidates

def process_query_batch(data, neighbours, query_vectors, candidate_buckets, indexes, offsets, freq_threshold, requested_amount, batch_process_start, process, max_cand_size, mem_tracking):
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
        candidates = get_candidates_for_query_vectorised(predicted_buckets, indexes, offsets)
        # Count occurrences of each element in cands_unf using np.bincount
        counts = np.bincount(candidates)
        # Get valid elements (those whose counts are greater than or equal to threshold)
        unique_candidates = np.where(counts >= freq_threshold)[0]
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
            final_neighbours, dist_comps, current_mem = reorder(data, query, np.array(unique_candidates, dtype=int), requested_amount, process, track_mem)
            query_end = time.time()
            memory = current_mem if current_mem > memory else memory
            del unique_candidates
            batch_results[i] = (final_neighbours, neighbours[i], dist_comps, (query_end-query_start) + base_time_per_query, ut.recall_single(final_neighbours, neighbours[i]))
    return batch_results, memory

def query_multiple_batched(data, index, vectors, neighbours, config: Config):
    '''
    Run multiple queries from a set of query vectors i.e. "Test" from the ANN benchmark datsets.
    '''
    ((indexes, offsets), models) = index
    all_bucket_sizes = np.zeros(shape = (config.r, config.b), dtype=np.uint32)
    for r, offset in enumerate(offsets):
        all_bucket_sizes[r] = np.diff(offset, prepend=0)

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
    process = psutil.Process(os.getpid())
    max_cand_size = 0
    with torch.no_grad():
        for batch_data, batch_indices in query_loader:
            batch_process_start = time.time()
            print(f"Processing one query batch")
            predicted_buckets_per_query = np.zeros((len(batch_data), len(models), config.m), dtype=np.int32)
            for i, model in enumerate(models):
                bucket_probabilities = torch.sigmoid(model(batch_data))
                _, candidate_buckets = torch.topk(bucket_probabilities, config.m, dim=1)
                predicted_buckets_per_query[:, i, :] = candidate_buckets
            batch_results, current_mem = process_query_batch(data, neighbours[batch_indices], batch_data, predicted_buckets_per_query, indexes, offsets, config.freq_threshold, config.nr_ann, batch_process_start, process, max_cand_size, config.mem_tracking)
            memory = current_mem if current_mem > memory else memory
            results[batch_idx] = batch_results
            batch_idx += 1

    flattened_results = [item for sublist in results for item in sublist]
    return flattened_results
    
def reorder(data, query_vector, candidates, requested_amount, process, track_mem = False):
    '''
    Do true distance calculations on the candidate set of vectors and return the top 'requested_amount' candidates.
    ''' 
    #TODO: check if it makes sense to sort candidates before fetching from memmap to improve access pattern
    # candidates = np.sort(candidates)

    #TODO: figure out new way to take these measurements as they cause slowdown
    # process = psutil.Process(os.getpid())

    mem = 0 if not track_mem else process.memory_full_info().uss / (1024 ** 2)
    if not isinstance(data, IndexPQ):
        search_space = np.ascontiguousarray(data[candidates].copy())
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