import logging
import numpy as np
import os
import psutil # type: ignore
import time
import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader

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
    # create lookup tensors for on gpu label retrieval
    neighbours_tensor = torch.from_numpy(neighbours).to(torch.int64).to(config.device)
    lookup = torch.from_numpy(index).to(torch.int64).to(config.device)
    all_losses = np.zeros(shape=config.epochs*config.iterations)
    current_epoch = 0
    process = psutil.Process(os.getpid())
    ram = 0
    vram = 0 
    load_balances = []

    if config.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(config.device)
    else:
        print(f"NO CUDA device detected - NOT tracking vram")

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
                s = time.time()
                batch_labels = ut.get_labels(neighbours_tensor[batch_indices], lookup, config.b, config.device)
                e = time.time()

                label_times.append(e-s)
                optimizer.zero_grad()
                logits = model(batch_data)
                loss = criterion(logits, batch_labels)

                loss.backward()
                optimizer.step()
                batch_count += 1
                epoch_loss_sum += loss.item()

                if config.mem_tracking:
                    current_mem = process.memory_full_info().uss / (1024 ** 2)
                    ram = current_mem if current_mem > ram else ram

                del loss, batch_data, batch_labels
            finish = time.time()
            elapsed = finish-start
            print(f"epoch {epoch} took {elapsed}")
            print(f"label making took (sum): {sum(label_times)}")
            print(f"epoch {epoch} loss = {epoch_loss_sum}", flush=True)
            all_losses[current_epoch] = epoch_loss_sum
            current_epoch += 1
        if ((epoch+1) * (i+1) < config.epochs*config.iterations and sample_size != train_size) or sample_size == train_size:
            logging.info("Reassigning vectors to new buckets")
            model.eval()
            model.to(config.device)
            reassign_loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False, num_workers=8)
            start = time.time()
            print(f"reassigning buckets using mode: {config.reass_mode}")
            if config.reass_mode == 0:
                reass_ram = reassign_0(model, index, bucket_sizes, config, reassign_loader)
            elif config.reass_mode == 1:
                reass_ram = reassign_1(model, index, bucket_sizes, config, reassign_loader)
            elif config.reass_mode == 2:
                reass_ram = reassign_2(model, index, bucket_sizes, config, reassign_loader)
            elif config.reass_mode == 3:
                reass_ram = reassign_3(model, index, bucket_sizes, config, reassign_loader)

            if config.mem_tracking:
                ram = reass_ram if reass_ram > ram else ram 

            lookup = torch.from_numpy(index).to(torch.int64).to(config.device)
            finish = time.time()
            elapsed = finish - start
            print(f"Reassigning took {elapsed:.2f} seconds", flush=True)

        if config.device.type == "cuda":
            vram = torch.cuda.max_memory_allocated(config.device)

        load_balances.append(1 / np.std(bucket_sizes))
        np.set_printoptions(threshold=6, suppress=True)
        print(f"index after iteration {i}: \r{index}", flush=True)

    # ut.make_loss_plot(config.lr, config.iterations, config.epochs, config.k, config.b, config.experiment_name, all_losses, config.shuffle, config.reass_mode)
    return ram, vram, load_balances


def reassign_0(model, index, bucket_sizes, config: Config, reassign_loader):
    '''
    Baseline implementation from the original BLISS code from <insert github link>. 
    Used for reference to compare to our improved version.
    '''
    N = len(index)
    candidate_buckets = np.zeros(shape = (N, config.k), dtype=np.uint32)
    memory_usage=0
    ut.get_all_topk_buckets(reassign_loader, config.k, candidate_buckets, model, 0, config.device)

    if config.mem_tracking:
        process = psutil.Process(os.getpid())
        memory_usage = process.memory_full_info().uss / (1024 ** 2)

    bucket_sizes[:] = 0
    for i in range(N):
        ut.reassign_vector_to_bucket(index, bucket_sizes, candidate_buckets[i], i)
    return memory_usage

def reassign_1(model, index, bucket_sizes, config: Config, reassign_loader):
    '''
    Reassign all items in the dataset to new buckets. This function is invoked after every x epochs of training, to improve assignments.
    To obtain a new bucket for a single vector, the model is used to predict the best buckets for that vector and the least occupied of k buckets
    is chosen as the new bucket. After all vectors are reassigned, new ground truth labels are generated to continue training.
    '''  
    bucket_sizes[:] = 0  
    memory = 0
    with torch.no_grad():
        for batch_data, batch_indices in reassign_loader:

            batch_data = batch_data.to(config.device)
            bucket_probabilities = torch.sigmoid(model(batch_data))
            bucket_probabilities_cpu = bucket_probabilities.cpu()

            _, candidate_buckets = torch.topk(bucket_probabilities_cpu, config.k, dim=1)
            candidate_buckets = candidate_buckets.numpy()
            if config.mem_tracking:
                    process = psutil.Process(os.getpid())
                    current_mem = process.memory_full_info().uss / (1024 ** 2)
                    memory = current_mem if current_mem > memory else memory
            for i, item_index in enumerate(batch_indices):
                ut.reassign_vector_to_bucket(index, bucket_sizes, candidate_buckets[i], item_index)
    return memory

def reassign_2(model, index, bucket_sizes, config: Config, reassign_loader):
    '''
    Variation of baseline reassignment, where we get all of the topk buckets beforehand, but reassign in chunks
    instead of sequentially.
    '''  
    N = len(index)
    candidate_buckets = np.zeros(shape = (N, config.k), dtype=np.uint32)
    memory = 0
    ut.get_all_topk_buckets(reassign_loader, config.k, candidate_buckets, model, 0, config.device)

    if config.mem_tracking:
        process = psutil.Process(os.getpid())
        memory = process.memory_full_info().uss / (1024 ** 2)

    chunk_size = config.reass_chunk_size
    bucket_sizes[:] = 0
    for i in range(0, N, chunk_size):
        topk_per_vector = candidate_buckets[i : min(i + chunk_size, N)]
        current_mem = ut.assign_to_buckets_vectorised(bucket_sizes, N, index, chunk_size, i, topk_per_vector)
        if config.mem_tracking:
            memory = current_mem if current_mem > memory else memory
    return memory
    
def reassign_3(model, index, bucket_sizes,  config: Config, reassign_loader):
    '''
    Combination of baseline and our own reassignment, where we get the topk buckets in batches and also
    reassign in chunks instead of sequentially.
    ''' 
    offset = 0
    N = len(index)
    bucket_sizes [:] = 0
    batch_size = config.batch_size
    memory = 0 
    with torch.no_grad():
        for batch_data, _ in reassign_loader:
            topk_per_vector = ut.get_topk_buckets_for_batch(batch_data, config.k, model, config.device)
            topk_per_vector.numpy()
            current_memory = ut.assign_to_buckets_vectorised_rm3(bucket_sizes, N, index, batch_size, offset, topk_per_vector)
            if config.mem_tracking:
                memory = current_memory if current_memory > memory else memory
            offset += batch_size
            del batch_data, topk_per_vector
    return memory