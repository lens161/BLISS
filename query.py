import numpy as np
import torch
from functools import partial
from multiprocessing import Pool
import sys
import time
import os
from utils import get_nearest_neighbours_in_different_dataset

def query(data, index, query_vector, neighbours, m, freq_threshold, requested_amount):
    '''
    Query the index for a single vector. Get the candidate set of vectors predicted by each of the R models.
    Then, filter the candidate set based on the frequency threshold.
    If the number of candidates exceeds the requested amount, reorder using true distance computations.
    Then return the remaining set of candidates.
    '''
    inverted_indexes, models = index
    candidates = {}
    for i in range(len(models)):
        model = models[i]
        model.eval()
        index = inverted_indexes[i]
        get_candidates_from_model(model, index, candidates, query_vector, m) 
    final_results = [key for key, value in candidates.items() if value >= freq_threshold]
    # print (f"final results = {len(final_results)}")
    if len(final_results) <= requested_amount:
        return final_results, 0, recall_single(final_results, neighbours)
    else:
        final_neighbours, dist_comps = reorder(data, query_vector, np.array(final_results, dtype=int), requested_amount)
        return final_neighbours, dist_comps, recall_single(final_neighbours, neighbours)

def get_candidates_from_model(model, index, candidates, query_vector, m):
    '''
    For a given model, do a forward pass for query_vector to get the top m buckets for this query. Update the candidate set with all vectors found in those buckets.
    '''
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

def process_query_chunk(chunk, data, index, m, threshold, requested_amount):
    '''
    Helper function to unpack vectors and neighbours from chunk, as partial application can only take one new argument.
    '''
    vectors, neighbours = chunk
    return query_multiple(data, index, vectors, neighbours, m, threshold, requested_amount, parallel=True)

def query_multiple_parallel(data, index, vectors, neighbours, m, threshold, requested_amount, num_workers):
    '''
    Query in parallel. Tasks of chunk_size are created and a process pool is used to process tasks until all tasks have been completed.
    Each task is sent to query_multiple to be processed as if it was a single-threaded task.
    '''
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
    '''
    Run multiple queries from a set of query vectors i.e. "Test" from the ANN benchmark datsets.
    '''
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
    
def reorder(data, query_vector, candidates, requested_amount):
    '''
    Do true distance calculations on the candidate set of vectors and return the top 'requested_amount' candidates.
    ''' 
    sp_index = []
    search_space = data[candidates]
    dist_comps = len(search_space)
    for i in range(len(search_space)):
        sp_index.append(candidates[i])
    # print(f"search_space = {search_space}")
    query_vector = query_vector.reshape(1, -1)
    neighbours = get_nearest_neighbours_in_different_dataset(search_space, query_vector, requested_amount)
    neighbours = neighbours[0].tolist()
    final_neighbours = []
    for i in neighbours:
        final_neighbours.append(sp_index[i])
    return final_neighbours, dist_comps

def recall(results, neighbours):
    '''
    Calculate mean recall for a set of queries.
    '''
    recalls = []
    for ann, nn in zip(results, neighbours):
        recalls.append(recall_single(ann, nn))
    return np.mean(recalls)

def recall_single(results, neighbours):
    '''
    Calculate recall for an individual query.
    '''
    return len(set(results) & set(neighbours))/len(neighbours)