import math
import numpy as np
import time
import torch

def make_sparse_ground_truth_labels(B, neighbours, index, sample_size, device):
    start = time.time()

    # initialise empty sparse tensor
    indices = torch.empty(2, 0, dtype=torch.int32)
    values = torch.empty(0, dtype=torch.int32)
    labels = torch.sparse_coo_tensor(indices, values, size=(sample_size, B), dtype=bool)

    # get new values
    vectors = np.concatenate([np.full(len(n), i) for i, n in enumerate(neighbours)])
    buckets = np.concatenate([index[n] for n in neighbours])
    indices = np.array([vectors, buckets])
    values = np.ones(len(indices[0]))

    # update sparse tensor
    labels = torch.sparse_coo_tensor(indices, values, (sample_size, B))

    print(f"making ground truch labels {time.time()-start}")
    return labels

# NOTE: not confirmed to be working, sparse labels project was abandoned
def make_sparse_ground_truth_labels(B, neighbours, index, sample_size, batch_size):
    start = time.time()

    nr_batches = math.ceil(sample_size / batch_size)
    # print(f"Expected nr of batches: {nr_batches}")
    # make prechunked list with a sparse tensor for each batch
    batch_labels_list = [torch.sparse_coo_tensor(torch.empty(2, 0, dtype=torch.int32), 
                                                 torch.empty(0, dtype=torch.int32), 
                                                 (batch_size, B), dtype=torch.bool) 
                        for _ in range(nr_batches)]
    # print(f"Made list of empty sparse tensors with size {len(batch_labels_list)}")

    # fill each position in the list with a sparse tensor
    start = 0
    for batch_idx in range(nr_batches):
        end = min(start+batch_size, sample_size)
        actual_batch_size = end-start
        # print(f"Grabbing neighbours chunk from {start} to {end}: {neighbours[start:end, :]}")
        vectors = np.concatenate([np.full(len(n), i+start) for i, n in enumerate(neighbours[start:end, :])])
        buckets = np.concatenate([index[n] for n in neighbours[start:end, :]])
        indices = np.array([vectors, buckets])
        values = np.ones(len(indices[0]))
        chunk_labels = torch.sparse_coo_tensor(indices, values, (actual_batch_size, B))
        # print(chunk_labels)
        batch_labels_list[batch_idx] = chunk_labels
        start += actual_batch_size

    print(f"making ground truth labels {time.time()-start}")
    return batch_labels_list

def make_ground_truth_labels(B, neighbours, index, sample_size):
    '''
    Create ground truth labels for training sample, based on the set of nearest neighbours of each training vector. 
    A label is a B-dimensional vector, where each digit is either 0 (false) if that bucket does not contain any
    nearest neighbours of a vector, and 1 (true) if the bucket contains at least one nearest neighbour of that vector.
    '''
    # start = time.time()
    labels = np.zeros((sample_size, B), dtype=bool)
    # for each vector i create an array of amount of neighbours
    vectors = np.concatenate([np.full(len(n), i) for i, n in enumerate(neighbours)])
    # build column indices by applying the mapping to each neighbour array.
    buckets = np.concatenate([index[n] for n in neighbours])
    # Use advanced indexing to set the corresponding entries to True.
    labels[vectors, buckets] = True
    # for i in range(sample_size):
    #     # buckets = index[neighbours[i]]
    #     # labels1[i, buckets] = 1
    #     for neighbour in neighbours[i]:
    #         bucket = index[neighbour]
    #         labels2[i, bucket] = True
    # print(f"making ground truch labels {time.time()-start}")
    # if device != torch.device("cpu"):
    #     labels = torch.from_numpy(labels).to(torch.float32)
    return torch.from_numpy(labels).float()