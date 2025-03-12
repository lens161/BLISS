from config import Config
from bliss import run_bliss
from utils import get_best_device
import numpy as np
import time 
import csv
import os
import pandas as pd
import matplotlib.pyplot as plt # type: ignore

def run_experiment(config: Config, mode = 'query'):
    # TO-DO: 
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
        time_per_r, build_time, memory_usage = run_bliss(config, mode=mode, experiment_name=experiment_name)
        stats.append({'R':r, 'k':k, 'epochs_per_it':epochs, 'iterations':iterations, 'build_time':build_time, 
                      'mem':memory_usage, 'shuffle':config.shuffle, 'global_reass': config.global_reass})
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
        plt.ylabel("Recall")
        plt.title(f"Distance Computations vs Recall R={r} k={k} m={m}")
        plt.grid(True)
        foldername = f"results/{experiment_name}"
        if not os.path.exists("results"):
            os.mkdir("results")
        if not os.path.exists(f"results/{experiment_name}"):
            os.mkdir(foldername)
        df.to_csv(f"{foldername}/r{r}_k{k}_m{m}_qps{qps:.2f}_avg_rec{avg_recall:.3f}_shf={config.shuffle}_gr={config.global_reass}.csv", index=False)
        # plt.show()
        plt.savefig(f"{foldername}/r{r}_k{k}_m{m}_qps{qps:.2f}_avg_rec{avg_recall:.3f}_shf={config.shuffle}_gr={config.global_reass}.png", dpi=300)

    return experiment_name, avg_recall, total_query_time, results

if __name__ == "__main__":
    configs_q = [] # configs for building the index
    configs_b = [] # configs for querying

    # range_M = 10
    # range_K = 2
    range_threshold = 2
    k_values = [2]
    m_values = [2, 5, 10, 15, 20]

    EXP_NAME = "global_vs_batched_reass"

    # add all dataset names that the experiments should be run on
    datasets = ["sift-128-euclidean", 
                # "glove-100-angular",
                 ]
    
    for dataset in datasets:
        conf_shuff = Config(dataset_name=dataset, batch_size=2048, r=2, epochs=2, iterations=2, shuffle=False)
        # conf_noshuff = Config(dataset_name=dataset, batch_size=2048, r=1, epochs=1, iterations=1)
        # conf_global = Config(dataset_name=dataset, batch_size=2048, r=1, epochs=1, iterations=1, global_reass=True)
        configs_b.append(conf_shuff)
        # configs_b.append(conf_noshuff)
        # configs_b.append(conf_global)s
        configs_q.append(conf_shuff)
        # configs_q.append(conf_noshuff)
        # configs_q.append(conf_global)

    # for dataset in datasets:
    #     conf_shuff = Config(dataset_name=dataset, k=2, r=4, batch_size=2048, b= 4096, epochs=5, iterations=4, shuffle=True)
    #     conf_noshuff = Config(dataset_name=dataset, k=2, r=4, batch_size=2048, b= 4096, epochs=5, iterations=4)
    #     conf_gr = Config(dataset_name=dataset, k=2, r=4, batch_size=2048, b= 4096, epochs=5, iterations=4, global_reass=True)
    #     configs_b.append(conf_shuff)
    #     configs_b.append(conf_noshuff)
    #     configs_b.append(conf_gr)
    #     for m in m_values:
    #         conf_q_shuff = Config(dataset_name=dataset, k=2, r=4, batch_size=2048, b= 4096, epochs=5, iterations=4, m=m, shuffle=True)
    #         conf_q_noshuff = Config(dataset_name=dataset, k=2, r=4, batch_size=2048, b= 4096, epochs=5, iterations=4, m=m)
    #         conf_q_gr = Config(dataset_name=dataset, k=2, r=4, batch_size=2048, b= 4096, epochs=5, iterations=4, m=m, global_reass=True)
    #         configs_q.append(conf_q_shuff)
    #         configs_q.append(conf_q_noshuff)
    #         configs_q.append(conf_q_gr)

    build_multiple_indexes_exp(EXP_NAME, configs_b)
    # run_multiple_query_exp(EXP_NAME, configs_q)
 