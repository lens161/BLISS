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
        r, k, epochs, iterations, b, lr, reass_chunk_size, batch_size, reass_mode, freq_threshold = config.r, config.k, config.epochs, config.iterations, config.b, config.lr, config.reass_chunk_size, config.batch_size, config.reass_mode, config.freq_threshold

        train_time, final_assign_time, build_time, memory_final_assignment, memory_training, normalised_entropy, index_sizes_total, model_sizes_total = run_bliss(config, mode=mode, experiment_name=experiment_name)

                        # Hyperparameters:
        stats.append({'R':r, 'k':k, 'epochs_per_it':epochs, 'iterations':iterations, 
                      'b':b, 'lr':lr,'batch_size':batch_size, 'freq_threshold':freq_threshold,
                      'reass_mode': reass_mode, 'reass_chunk_size':reass_chunk_size,
                        # Measurements/Results:
                      'build_time':build_time, 'train_time_per_r':train_time, 'final_assign_time_per_r':final_assign_time,
                      'mem_training':memory_training, 'mem_final_ass':memory_final_assignment, 'load_balance':normalised_entropy, 
                      'index_sizes_total': index_sizes_total,'model_sizes_total': model_sizes_total})
    df = pd.DataFrame(stats)
        
    foldername = f"results/{experiment_name}"
    os.makedirs(foldername, exist_ok=True)
    csv_path = os.path.join(foldername, f"{experiment_name}_build.csv")
    if os.path.isfile(csv_path):
        df.to_csv(csv_path, mode='a', header=False, index=False)
    else:
        df.to_csv(csv_path, mode='w', header=True, index=False)

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
    EXP_NAME = "check_new_model_names"

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
    
    logging.info("[Experiment] Experiments started")
    # check that datasize in config is set to correct value. (default = 1)
    # configs_b.append(Config(dataset_name="sift-128-euclidean", batch_size=2048, b=4096, r=2, epochs=2, iterations=2))
    configs_q.append(Config(dataset_name="sift-128-euclidean", batch_size=2048, b=4096, r=2, epochs=2, iterations=2,  mem_tracking=True))
    # configs_b.append(Config(dataset_name="sift-128-euclidean", batch_size=2048, b=4096, m=10, datasize=10))
    # configs_q.append(Config(dataset_name="bigann", batch_size=2048, b=4096, m=10, pq=True, datasize=10))
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