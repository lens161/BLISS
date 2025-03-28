import logging
import matplotlib.pyplot as plt # type: ignore
import os
import pandas as pd

from bliss_pq import run_bliss
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
                      'mem':memory_usage, 'load_balance':load_balance, 'shuffle':config.shuffle, 'global_reass': config.global_reass})
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
        df.to_csv(f"{foldername}/r{r}_k{k}_m{m}_qps{qps:.2f}_avg_rec{avg_recall:.3f}_shf={config.shuffle}_gr={config.global_reass}_nr_ann={config.nr_ann}.csv", index=False)
        plt.savefig(f"{foldername}/r{r}_k{k}_m{m}_qps{qps:.2f}_avg_rec{avg_recall:.3f}_shf={config.shuffle}_gr={config.global_reass}_nr_ann={config.nr_ann}.png", dpi=300)

    return experiment_name, avg_recall, total_query_time, results

if __name__ == "__main__":
    configs_q = [] # configs for building the index
    configs_b = [] # configs for querying
    # range_M = 10
    # range_K = 2
    range_threshold = 2
    k_values = [2]
    m_values = [2]
    EXP_NAME = "test_refactor_p4"

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
                # "glove-100-angular",
                "sift-128-euclidean"
                 ]
    
    logging.info("[Experiment] Experiments started")
        # check that datasize in config is set to correct value. (default = 1)
    for dataset in datasets:
        conf_4 = Config(dataset_name=dataset, batch_size=2048, b= 1024, epochs=2, iterations=2)
        # conf_8 = Config(dataset_name=dataset, batch_size=2048, b=8192, datasize=10)
        configs_b.append(conf_4)
        # configs_b.append(conf_8)
        for m in m_values:
            conf_q4 = Config(dataset_name=dataset, batch_size=2048, m=m, b=1024, epochs=2, iterations=2)
            # conf_q8 = Config(dataset_name=dataset, batch_size=2048, m=m, b=8192, datasize=10)
            configs_q.append(conf_q4)
            # configs_q.append(conf_q8)
    
    logging.info(f"[Experiment] Building indexes")
    # build_multiple_indexes_exp(EXP_NAME, configs_b)
    logging.info(f"[Experiment] Starting query experiments")
    run_multiple_query_exp(EXP_NAME, configs_q)