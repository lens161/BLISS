import time
import torch

import utils as ut
from build import build_index
from config import Config
from query import query_multiple, query_multiple_batched, query_multiple_batched_twostep, query_multiple_twostep

def run_bliss(config: Config, mode, experiment_name):
    '''
    Run the BLISS algorithm. Mode determines whether an index is built or whether inference is run on an existing index.
    '''

    config.experiment_name = experiment_name
    print(f"Using device: {config.device}")

    dataset = ut.get_dataset_obj(config.dataset_name, config.datasize)
    SIZE = dataset.nb
    DIM = dataset.d
    dataset.prepare()
    # set b if it wasn't already set in config
    if config.b == 0:
        config.b = ut.get_B(SIZE)
    if mode == 'build':
        index, train_time, final_assign_time, build_time, memory_final_assignment, memory_training, load_balance, index_sizes_total, model_sizes_total, load_balances = build_index(dataset, config)
        return train_time, final_assign_time, build_time, memory_final_assignment, memory_training, load_balance, index_sizes_total, model_sizes_total, load_balances
    elif mode == 'query':       
        print("Loading models for inference")
        index = ut.load_indexes_and_models(config, SIZE, DIM, config.b)
        print("Reading query vectors and ground truths")
        data, test, neighbours, using_memmap = ut.load_data_for_inference(dataset, config, SIZE)
 
        print(f"creating tensor array from Test")
        test = torch.from_numpy(test).to(torch.float32)
        
        print("Starting inference")
        start = time.time()
        qstart = time.time()
        if config.query_batched and config.query_twostep:
            results = query_multiple_batched_twostep(data, index, test, neighbours, config, using_memmap)
        elif config.query_batched:
            results = query_multiple_batched(data, index, test, neighbours, config, using_memmap)
        elif config.query_twostep:
            results = query_multiple_twostep(data, index, test, neighbours, config, using_memmap)
        else:
            results = query_multiple(data, index, test, neighbours, config, using_memmap)
        print(f"querying took {time.time()-qstart}")
        end = time.time()

        total_query_time = end - start

        anns = [t[0] for t in results]
        RECALL = ut.recall(anns, neighbours)
        print(f"RECALL = {RECALL}", flush=True)
        return RECALL, results, total_query_time
