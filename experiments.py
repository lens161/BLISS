import logging
import matplotlib.pyplot as plt # type: ignore
import os
import pandas as pd

from bliss import run_bliss
from config import Config


def run_experiment(config: Config, mode = 'query'):
    # TODO: 
    # - seperate query statistics from building statistics
    avg_recall, stats, total_query_time = run_bliss(config, mode= mode)
    return total_query_time, avg_recall, stats
def build_multiple_indexes_exp(experiment_name, configs):
    mode = 'build'
    stats = []
    for config in configs:
        r = config.r
        k = config.k
        epochs = config.epochs
        iterations = config.iterations
        time_per_r, build_time, memory_usage, load_balance = run_bliss(config, mode=mode, experiment_name=experiment_name)
        stats.append({'R':r, 'k':k, 'epochs_per_it':epochs, 'iterations':iterations, 'build_time':build_time, 
                      'mem':memory_usage, 'load_balance':load_balance, 'shuffle':config.shuffle, 'reass_mode': config.reass_mode})
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
        r = config.r
        k = config.k
        m =config.m
        results = []
        avg_recall, stats, total_query_time = run_bliss(config, mode=mode, experiment_name=experiment_name)
        print(f"avg recall = {avg_recall}")
        for (anns, dist_comps, elapsed, recall) in stats:
            results.append({'ANNs': anns, 
                            'distance_computations': dist_comps, 
                            'elapsed': elapsed,
                            'recall': recall})
        qps = len(stats)/total_query_time
        df = pd.DataFrame(results)
        plt.figure(figsize=(8, 5))
        plt.scatter(df['distance_computations'], df['recall'], color='blue', s=20)
        plt.xlabel("Distance Computations")
        plt.ylabel(f"Recall ({config.nr_ann}@{config.nr_ann})")
        plt.title(f"Distance Computations vs Recall R={r} k={k} m={m}")
        plt.grid(True)
        foldername = f"results/{experiment_name}"
        if not os.path.exists("results"):
            os.mkdir("results")
        if not os.path.exists(f"results/{experiment_name}"):
            os.mkdir(foldername)
        df.to_csv(f"{foldername}/r{r}_k{k}_m{m}_qps{qps:.2f}_avg_rec{avg_recall:.3f}_bs={config.batch_size}_reass={config.reass_mode}_nr_ann={config.nr_ann}_lr={config.lr}.csv", index=False)
        plt.savefig(f"{foldername}/r{r}_k{k}_m{m}_qps{qps:.2f}_avg_rec{avg_recall:.3f}_bs={config.batch_size}_reass={config.reass_mode}_nr_ann={config.nr_ann}_lr={config.lr}.png", dpi=300)

    return experiment_name, avg_recall, total_query_time, results

if __name__ == "__main__":
    configs_q = [] # configs for building the index
    configs_b = [] # configs for querying
    # range_M = 10
    # range_K = 2
    range_threshold = 2
    k_values = [2]
    m_values = [15]
    reass_modes = [2]
    batch_sizes = [1024, 2048, 5000]
    EXP_NAME = "check_refact_reass_1-2"

    if not os.path.exists("logs"):
        os.mkdir("logs")

    logging.basicConfig(
        filename=f'logs/{EXP_NAME}.log',               # Specify the log file name
        level=logging.INFO,               # Set the logging level to INFO
        format='%(asctime)s - %(levelname)s - %(message)s'  # Define the log message format
    )

    # add all dataset names that the experiments should be run on
    datasets = [
                # "bigann",
                "glove-100-angular",
                # "sift-128-euclidean"
                 ]
    
    logging.info("[Experiment] Experiments started")
        # check that datasize in config is set to correct value. (default = 1)
    for dataset in datasets:
        for bs in batch_sizes:
            conf = Config(dataset_name=dataset, batch_size=bs, b=4096)
            configs_b.append(conf)
            if bs == 5000:
                conf1 = Config(dataset_name=dataset, batch_size=bs, b=4096, lr=0.01)
                configs_b.append(conf1)
        # for rm in reass_modes:
            # for m in m_values:
            conf_q = Config(dataset_name=dataset, batch_size=bs, b=4096)
            configs_q.append(conf_q)
            if bs == 5000:
                conf1 = Config(dataset_name=dataset, batch_size=bs, b=4096, lr=0.01)
                configs_q.append(conf1)
    
    logging.info(f"[Experiment] Building indexes")
    build_multiple_indexes_exp(EXP_NAME, configs_b)
    logging.info(f"[Experiment] Starting query experiments")
    run_multiple_query_exp(EXP_NAME, configs_q)