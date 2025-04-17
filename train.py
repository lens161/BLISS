import logging
import numpy as np
import os
import psutil # type: ignore
import statistics
import time
import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader
from pympler import asizeof

import utils as ut
from config import Config


def train_model(model, dataset, index, sample_size, bucket_sizes, neighbours, r, train_size, config: Config):
    '''
    Train a BLISS model. BCEWithLogitsLoss is used as criterion as the labels are multi-hot encoded.
    Vectors are passed through the model in batches and the model learns to minimize the loss.
    After every few epochs (set in config), vectors are reassigned to new buckets based on the current state of the model.
    For million-scale data, as all data is trained on, the last reassignment will be used as the final assignment.
    For larger datasets, the last reassignment is skipped, as training is only done in a sample but all remaining data needs to
    be assigned immediately after training the model.
    '''
    model.to(config.device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    g = torch.Generator()
    g.manual_seed(r)
    train_loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, num_workers=8, generator=g)
    all_losses = np.zeros(shape=config.epochs*config.iterations)
    current_epoch = 0
    indices = torch.empty(2, 0, dtype=torch.int32)
    values = torch.empty(0, dtype=torch.int32)
    process = psutil.Process(os.getpid())
    memory_training = 0
    sparse_labels = torch.sparse_coo_tensor(indices, values, size=(sample_size, config.b), dtype=bool)
    for i in range(config.iterations):
        model.train() 
        for epoch in range(config.epochs):
            epoch_loss_sum = 0
            batch_count = 0
            print(f"training epoch ({i}, {epoch})")
            logging.info(f"Training epoch ({i}, {epoch})")
            start = time.time()
            label_times = []
            for batch_data, batch_indices in train_loader:
                batch_data = batch_data.to(config.device)
                # batch_labels = batch_labels.to(config.device)
                s = time.time()
                batch_labels = ut.make_ground_truth_labels(config.b, neighbours[batch_indices], index, len(batch_data)).to(config.device)
                e = time.time()
                memory_current = process.memory_full_info().uss / (1024 ** 2)
                memory_training = memory_current if memory_current>memory_training else memory_training
                label_times.append(e-s)
                # if isinstance(batch_labels, torch.Tensor) and batch_labels.is_sparse:
                #     batch_labels = batch_labels.to_dense()
                optimizer.zero_grad()
                probabilities = model(batch_data)
                loss = criterion(probabilities, batch_labels)
                loss.backward()
                optimizer.step()
                batch_count += 1
                epoch_loss_sum += loss.item()
                del loss, batch_data, batch_labels
            finish = time.time()
            elapsed = finish-start
            print(f"epoch {epoch} took {elapsed}")
            print(f"label making took (sum): {sum(label_times)}")
            print(f"epoch {epoch} loss = {epoch_loss_sum}", flush=True)
            all_losses[current_epoch]
            current_epoch += 1
        # if config.reass_mode == 0:
            # reassign_base(model, dataset, index, neighbours, bucket_sizes, config)
        if ((epoch+1) * (i+1) < config.epochs*config.iterations and sample_size != train_size) or sample_size == train_size:
            reassign_buckets(model, dataset, index, bucket_sizes, sample_size, neighbours, config)
        torch.cuda.empty_cache()
        np.set_printoptions(threshold=6, suppress=True)
        print(f"index after iteration {i}: \r{index}", flush=True)

    # ut.make_loss_plot(config.lr, config.iterations, config.epochs, config.k, config.b, config.experiment_name, all_losses, config.shuffle, config.reass_mode)
    return memory_training

def reassign_buckets(model, dataset, index, bucket_sizes, sample_size, neighbours, config: Config):
    '''
    Reassign all items in the dataset to new buckets. This function is invoked after every x epochs of training, to improve assignments.
    To obtain a new bucket for a single vector, the model is used to predict the best buckets for that vector and the least occupied of k buckets
    is chosen as the new bucket. After all vectors are reassigned, new ground truth labels are generated to continue training.
    '''
    logging.info("Reassigning vectors to new buckets (improved version)")
    sample_size, _ = np.shape(dataset.data)
    model.to(config.device)
    model.eval()
    reassign_loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False, num_workers=8)
    
    bucket_sizes[:] = 0  

    start = time.time()
    with torch.no_grad():
        for batch_data, batch_indices in reassign_loader:

            batch_data = batch_data.to(config.device)
            bucket_probabilities = torch.sigmoid(model(batch_data))
            bucket_probabilities_cpu = bucket_probabilities.cpu()

            _, candidate_buckets = torch.topk(bucket_probabilities_cpu, config.k, dim=1)
            candidate_buckets = candidate_buckets.numpy()
            
            for i, item_index in enumerate(batch_indices):
                ut.reassign_vector_to_bucket(index, bucket_sizes, candidate_buckets, i, item_index)

    finish = time.time()
    elapsed = finish - start
    process = psutil.Process(os.getpid())
    mem_usage = process.memory_full_info().uss / (1024 ** 2)
    ut.log_mem(f"improved_reassign_buckets", mem_usage, config.memlog_path)

    print(f"Memory usage (improved reassign): {mem_usage:.2f} MB")
    print(f"Reassigning took {elapsed:.2f} seconds", flush=True)
    
    # new_labels = ut.make_ground_truth_labels(config.b, neighbours, index, sample_size)
    # dataset.labels = new_labels
    model.to(config.device)

def reassign_base(model, dataset, index, neighbours, bucket_sizes, config: Config):
    '''
    Baseline implementation from the original BLISS code from <insert github link>. 
    Used for reference to compare to our improved version.
    '''
    model.eval()
    model.to(config.device)
    data_loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False, num_workers=8)
    N = len(index)
    
    candidate_buckets = np.zeros(shape = (N, config.k))

    start = time.time() 
    ut.get_all_topk_buckets(data_loader, config.k, candidate_buckets, model, 0, config.device)
    # concatenate all predictions along the 0th dimension -> create tensor of predictions per vector of shape(N, B)
    # all_predictions = torch.cat(all_predictions, dim=0) 
    process = psutil.Process(os.getpid())
    mem_usage = process.memory_full_info().uss / (1024 ** 2)
    print(f"basline reass memory usage: {mem_usage:.2f} MB")
    ut.log_mem("reassign_base", mem_usage, config.memlog_path)

    bucket_sizes[:] = 0
    for i in range(N):
        ut.reassign_vector_to_bucket(index, bucket_sizes, candidate_buckets, i, i)
    
    finish = time.time()
    elapsed = finish - start

    print(f"reassigning took {elapsed}")
    # new_labels = ut.make_ground_truth_labels(config.b, neighbours, index, N)
    # dataset.labels = new_labels
    print("New bucket sizes:", bucket_sizes)


# def reassign_buckets_vectorized(model, dataset, index, bucket_sizes, sample_size, neighbours,device, config: Config):
#     '''
#     Alternative version of reassign_buckets, where buckets are updated in batches.
#     Note: does not give 100% guarantee that the least occupied bucket is selected for each vector, if multiple vectors
#     in a batch are assigned to the same bucket.
#     '''
#     sample_size, _ = np.shape(dataset.data)
#     model.to(config.device)
#     model.eval()
#     reassign_loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False, num_workers=8)
    
#     # create auxiliary tensor representing bucket_sizes
#     bucket_sizes_t = torch.zeros(config.B, device=device, dtype=torch.int32)
    
#     global_item_index = 0 # index from list "index" on cpu
#     start = time.time()
#     with torch.no_grad():
#         for batch_data, _ in reassign_loader:
#             batch_data = batch_data.to(config.device)
#             bucket_probabilities = torch.sigmoid(model(batch_data))
#             _, candidate_indices = torch.topk(bucket_probabilities, config.k, dim=1)
#             # get current sizes for candidate buckets (broadcast bucket_sizes_t)
#             candidate_sizes = bucket_sizes_t[candidate_indices]  # shape: (batch_size, k)
#             #for each vector, choose the candidate with the smallest bucket count
#             chosen_candidate_idx = torch.argmin(candidate_sizes, dim=1)
#             #get the chosen buckets for batch
#             chosen_buckets = candidate_indices[torch.arange(candidate_indices.size(0)), chosen_candidate_idx] #shape(batch_size, 1)
            
#             # batched update bucket sizes
#             ones = torch.ones_like(chosen_buckets, dtype=torch.int32, device=config.device)
#             bucket_sizes_t = bucket_sizes_t.scatter_add(0, chosen_buckets, ones)
            
#             # send chosen buckets to cpu
#             chosen_buckets_cpu = chosen_buckets.cpu().numpy()
#             for bucket in chosen_buckets_cpu:
#                 index[global_item_index] = bucket
#                 global_item_index += 1
#     end = time.time()
    
#     #update global bucket_sizes list
#     bucket_sizes[:] = bucket_sizes_t.cpu().numpy()
    
#     process = psutil.Process(os.getpid())
#     mem_usage = process.memory_info().rss
#     print(f"Memory usage: {mem_usage / (1024 ** 2):.2f} MB")
#     print(f"reassigning vect. took: {end - start:.4f} seconds")
    
#     new_labels = ut.make_ground_truth_labels(config.b, neighbours, index, sample_size, config.device)
#     dataset.labels = new_labels