import logging
import matplotlib.pyplot as plt # type: ignore
import numpy as np
import os
import pandas as pd

from bliss import run_bliss
from config import Config
from make_plots import make_plots


def run_experiment(config: Config, mode = 'query'):
    # TODO: 
    # - seperate query statistics from building statistics
    avg_recall, stats, total_query_time = run_bliss(config, mode= mode)
    return total_query_time, avg_recall, stats

def build_multiple_indexes_exp(experiment_name, configs):
    mode = 'build'
    stats = []
    for config in configs:
        r, k, epochs, iterations = config.r, config.k, config.epochs, config.iterations
        time_per_r, build_time, memory_final_assignment, memory_training, load_balance, index_sizes_total, model_sizes_total = run_bliss(config, mode=mode, experiment_name=experiment_name)
        stats.append({'R':r, 'k':k, 'epochs_per_it':epochs, 'iterations':iterations, 'build_time':build_time, 
                      'mem_final_ass':memory_final_assignment, 'mem_training':memory_training, 'load_balance':load_balance, 
                      'batch_size':config.batch_size, 'reass_mode': config.reass_mode,
                      'index_sizes_total': index_sizes_total,'model_sizes_total': model_sizes_total})
        print(time_per_r)
    foldername = f"results/{experiment_name}"
    if not os.path.exists("results"):
        os.mkdir("results")
    if not os.path.exists(f"results/{experiment_name}"):
        os.mkdir(foldername)
    df = pd.DataFrame(stats)
    df.to_csv(f"{foldername}/{experiment_name}_build.csv", index=False)

def run_multiple_query_exp(experiment_name, configs):
    mode = 'query'
    for config in configs:
        individual_results = []
        avg_recall, stats, total_query_time = run_bliss(config, mode=mode, experiment_name=experiment_name)
        print(f"avg recall = {avg_recall}")
        for (anns, true_nns, dist_comps, elapsed, recall) in stats:
            individual_results.append({'ANNs': ','.join(map(str, anns)) if isinstance(anns, (list, np.ndarray)) else str(anns), 
                            'true_nns': ','.join(map(str, true_nns)) if isinstance(true_nns, (list, np.ndarray)) else str(true_nns),
                            'distance_computations': dist_comps, 
                            'elapsed': elapsed,
                            'recall': recall})
        qps = len(stats)/total_query_time
        individual_results_df = pd.DataFrame(individual_results)
        avg_results_and_params = pd.DataFrame([{'r': config.r, 'k': config.k, 'm': config.m, 'bs': config.batch_size, 'reass_mode': config.reass_mode, 
                                               'nr_ann': config.nr_ann, 'lr': config.lr, 'avg_recall': avg_recall, 'qps': qps}])
        foldername = f"results/{experiment_name}"
        if not os.path.exists("results"):
            os.mkdir("results")
        if not os.path.exists(f"results/{experiment_name}"):
            os.mkdir(foldername)
        with pd.HDFStore(f"{foldername}/r{config.r}_k{config.k}_m{config.m}_qps{qps:.2f}_avg_rec{avg_recall:.3f}_bs={config.batch_size}_reass={config.reass_mode}_nr_ann={config.nr_ann}_lr={config.lr}.h5", mode='w') as store:
            store.put('individual_results', individual_results_df, format='table')
            store.put('averages', avg_results_and_params, format='table')

    return experiment_name, avg_recall, total_query_time, individual_results, avg_results_and_params

if __name__ == "__main__":
    configs_q = [] # configs for building the index
    configs_b = [] # configs for querying
    # range_M = 10
    # range_K = 2
    range_threshold = 2
    k_values = [2]
    m_values = [5, 10, 15]
    reass_modes = [0, 1, 2, 3]
    batch_sizes = [1024, 2048, 5000]
    EXP_NAME = "memory_pq_vs_nopq"

    if not os.path.exists("logs"):
        os.mkdir("logs")

    logging.basicConfig(
        filename=f'logs/{EXP_NAME}.log',                    # Specify the log file name
        level=logging.INFO,                                 # Set the logging level to INFO
        format='%(asctime)s - %(levelname)s - %(message)s'  # Define the log message format
    )

    # add all dataset names that the experiments should be run on
    datasets = [
                "bigann",
                # "glove-100-angular",
                # "sift-128-euclidean"
                 ]
    # sift
    # empty buckets: 983
    # precentage of all buckets: 0.239990234375
    # balance: 0.002716883572686393

    # bigann
    

    logging.info("[Experiment] Experiments started")
    # check that datasize in config is set to correct value. (default = 1)
    # configs_q.append(Config(dataset_name="bigann", batch_size=2048, b=4096, m=10, datasize=10))
    configs_q.append(Config(dataset_name="bigann", batch_size=2048, b=4096, m=10, pq=True, datasize=10))
    # configs_q.append(Config(dataset_name="bigann", batch_size=2048, b=4096, m=10, datasize=10))
    # configs_q.append(Config(dataset_name="bigann", batch_size=2048, b=4096, m=10, datasize=10,pq=True))
    # for dataset in datasets:
    #     for bs in batch_sizes:
    #         conf = Config(dataset_name=dataset, batch_size=bs, b=4096, datasize=10)
    #         configs_b.append(conf)
    #         if bs == 5000:
    #             conf1 = Config(dataset_name=dataset, batch_size=bs, b=4096, lr=0.01, datasize=10)
    #             configs_b.append(conf1)
    #     # for rm in reass_modes:
    #         # for m in m_values:
    #         # conf_q = Config(dataset_name=dataset, m=15, batch_size=bs, b=4096, datasize=10)
    #         # configs_q.append(conf_q)
    #         if bs == 5000:
    #             confq1 = Config(dataset_name=dataset, m=15, batch_size=bs, b=4096, lr=0.01, datasize=10)
    #             configs_q.append(confq1)
                
    print(f"EXPERIMENT: {EXP_NAME}")
    logging.info(f"[Experiment] Building indexes")
    # build_multiple_indexes_exp(EXP_NAME, configs_b)
    logging.info(f"[Experiment] Starting query experiments")
    run_multiple_query_exp(EXP_NAME, configs_q)

    make_plots(EXP_NAME)