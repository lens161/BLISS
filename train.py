import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader
import logging
from config import Config
import time
import statistics
import numpy as np
from utils import *
import psutil # type: ignore
from sklearn.utils import murmurhash3_32 as mmh3


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
    all_losses = []
    for i in range(config.iterations):
        model.train() 
        for epoch in range(config.epochs):
            epoch_losses = []
            print(f"training epoch ({i}, {epoch})")
            logging.info(f"Training epoch ({i}, {epoch})")
            start = time.time()
            for batch_data, batch_labels, _ in train_loader:
                batch_data = batch_data.to(config.device)
                batch_labels = batch_labels.to(config.device)
                optimizer.zero_grad()
                probabilities = model(batch_data)
                loss = criterion(probabilities, batch_labels)
                loss.backward()
                optimizer.step()
                epoch_losses.append(loss.item())
            finish = time.time()
            elapsed = finish-start
            print(f"epoch {epoch} took {elapsed}")
            avg_loss = statistics.mean(epoch_losses)
            print(f"epoch {epoch} avg. loss = {avg_loss}", flush=True)
            all_losses.append(avg_loss)
        if config.global_reass:
            global_reassign_buckets(model, dataset, index, neighbours, bucket_sizes, config)
        elif ((epoch+1) * (i+1) < config.epochs*config.iterations and sample_size != train_size) or sample_size == train_size:
            reassign_buckets(model, dataset, index, bucket_sizes, sample_size, neighbours, config)
        np.set_printoptions(threshold=6, suppress=True)
        print(f"index after iteration {i}: \r{index}", flush=True)

    make_loss_plot(config.lr, config.iterations, config.epochs, config.k, config.b, config.experiment_name, all_losses, config.shuffle, config.global_reass)
 
def reassign_buckets(model, dataset, index, bucket_sizes, sample_size, neighbours, config: Config):
    '''
    Reassign all items in the dataset to new buckets. This function is invoked after every x epochs of training, to improve assignments.
    To obtain a new bucket for a single vector, the model is used to predict the best buckets for that vector and the least occupied of k buckets
    is chosen as the new bucket. After all vectors are reassigned, new ground truth labels are generated to continue training.
    '''
    logging.info(f"Reassigning vectors to new buckets")
    sample_size, _ = np.shape(dataset.data)
    model.to("cpu")
    model.eval()
    reassign_loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=config.shuffle, num_workers=8)
    bucket_sizes[:] = 0

    start = time.time()
    with torch.no_grad():
        for batch_data, batch_labels, batch_indices in reassign_loader:
            batch_data = batch_data.to("cpu")
            bucket_probabilities = torch.sigmoid(model(batch_data))

            for probability_vector, idx in zip(bucket_probabilities, batch_indices):
                reassign_vector_to_bucket(probability_vector, index, bucket_sizes, config.k, idx)
                     
    finish = time.time()
    elapsed = finish - start
    process = psutil.Process(os.getpid())
    mem_usage = process.memory_info().rss / (1024 ** 2)
    log_mem(f"shuffle={config.shuffle}_reassign_buckets", mem_usage, config.memlog_path)

    print(f"Memory usage reassign batched: {mem_usage:.2f} MB")
    print(f"reassigning took {elapsed}", flush=True)
    new_labels = make_ground_truth_labels(config.b, neighbours, index, sample_size, config.device)
    dataset.labels = new_labels
    model.to(config.device)

def global_reassign_buckets(model, dataset, index, neighbours, bucket_sizes, config: Config):
    '''
    Alternative version of reassign_buckets, where first all bucket predictions are collected and then processed after.
    '''
    model.eval()
    data_loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False, num_workers=8)
    
    all_predictions = []

    start = time.time()
    with torch.no_grad():
        for batch_data, _, batch_indices in data_loader:
            batch_data = batch_data.to(config.device)
            probabilities = torch.sigmoid(model(batch_data))
            all_predictions.append(probabilities.cpu())
    
    # concatenate all predictions along the 0th dimension -> create tensor of predictions per vector of shape(N, B)
    all_predictions = torch.cat(all_predictions, dim=0) 
    process = psutil.Process(os.getpid())
    mem_usage = process.memory_info().rss / (1024 ** 2)
    print(f"global ress memory usage: {mem_usage:.2f} MB")
    log_mem("global_reassign_buckets", mem_usage, config.memlog_path)
    N = all_predictions.size(0)
    
    bucket_sizes[:] = 0
    
    for i in range(N):
        probs = all_predictions[i]
        _, candidate_buckets = torch.topk(probs, config.k)
        candidate_buckets = candidate_buckets.numpy()
        candidate_sizes = bucket_sizes[candidate_buckets]
        best_candidate = candidate_buckets[np.argmin(candidate_sizes)]
        index[i] = best_candidate
        bucket_sizes[best_candidate] += 1
    
    finish = time.time()
    elapsed = finish - start

    print(f"reassigning took {elapsed}")
    new_labels = make_ground_truth_labels(config.b, neighbours, index, N, config.device)
    dataset.labels = new_labels
    print("New bucket sizes:", bucket_sizes)


def reassign_buckets_vectorized(model, dataset, index, bucket_sizes, sample_size, neighbours,device, config: Config):
    '''
    Alternative version of reassign_buckets, where buckets are updated in batches.
    Note: does not give 100% guarantee that the least occupied bucket is selected for each vector, if multiple vectors
    in a batch are assigned to the same bucket.
    '''
    sample_size, _ = np.shape(dataset.data)
    model.to(config.device)
    model.eval()
    reassign_loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False, num_workers=8)
    
    # create auxiliary tensor representing bucket_sizes
    bucket_sizes_t = torch.zeros(config.B, device=device, dtype=torch.int32)
    
    global_item_index = 0 # index from list "index" on cpu
    start = time.time()
    with torch.no_grad():
        for batch_data, _ in reassign_loader:
            batch_data = batch_data.to(config.device)
            bucket_probabilities = torch.sigmoid(model(batch_data))
            _, candidate_indices = torch.topk(bucket_probabilities, config.k, dim=1)
            # get current sizes for candidate buckets (broadcast bucket_sizes_t)
            candidate_sizes = bucket_sizes_t[candidate_indices]  # shape: (batch_size, k)
            #for each vector, choose the candidate with the smallest bucket count
            chosen_candidate_idx = torch.argmin(candidate_sizes, dim=1)
            #get the chosen buckets for batch
            chosen_buckets = candidate_indices[torch.arange(candidate_indices.size(0)), chosen_candidate_idx]
            
            # batched update bucket sizes
            ones = torch.ones_like(chosen_buckets, dtype=torch.int32, device=config.device)
            bucket_sizes_t = bucket_sizes_t.scatter_add(0, chosen_buckets, ones)
            
            # send chosen buckets to cpu
            chosen_buckets_cpu = chosen_buckets.cpu().numpy()
            for bucket in chosen_buckets_cpu:
                index[global_item_index] = bucket
                global_item_index += 1
    end = time.time()
    
    #update global bucket_sizes list
    bucket_sizes[:] = bucket_sizes_t.cpu().numpy()
    
    process = psutil.Process(os.getpid())
    mem_usage = process.memory_info().rss
    print(f"Memory usage: {mem_usage / (1024 ** 2):.2f} MB")
    print(f"reassigning vect. took: {end - start:.4f} seconds")
    
    new_labels = make_ground_truth_labels(config.b, neighbours, index, sample_size, config.device)
    dataset.labels = new_labels