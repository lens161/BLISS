from config import Config
from bliss import run_bliss
from utils import get_best_device
import numpy as np
import time 
import csv
import os
import pandas as pd
import matplotlib.pyplot as plt # type: ignore
import datetime
import subprocess
import sys

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
        plt.savefig(f"{foldername}/r{r}_k{k}_m{m}_qps{qps:.2f}_avg_rec{avg_recall:.3f}_shf={config.shuffle}_gr={config.global_reass}.png", dpi=300)

    return experiment_name, avg_recall, total_query_time, results

def print_heartbeat():
    while(True):
        print(f"Process {os.getpid()} is still active. Timestamp: {datetime.datetime.now()}")
        time.sleep(300)

if __name__ == "__main__":
    configs_q = [] # configs for building the index
    configs_b = [] # configs for querying
    # range_M = 10
    # range_K = 2
    range_threshold = 2
    k_values = [2]
    m_values = [5, 10, 15, 20]
    EXP_NAME = "bigann_refactored_test"
    # add all dataset names that the experiments should be run on
    datasets = [
                # "bigann", 
                # "glove-100-angular",
                "sift-128-euclidean"
                 ]
    
    heartbeat_process = subprocess.Popen(
        ['python3', '-c', '''
import time
import datetime

# Function to print the message every 5 minutes
def subprocess_print():
    while True:
        print(f"This process is still running. Timestamp: {datetime.datetime.now()}", flush=True)
        time.sleep(300)  # Sleep for 5 minutes

# Start the function
subprocess_print()
        '''],
        stdout=sys.stdout,  # Redirect stdout to the main process stdout
        stderr=sys.stderr,  # Optionally redirect stderr to the main process stderr
    )

    try:
        # check that datasize in config is set to correct value. (default = 1)
        for dataset in datasets:
            conf_4 = Config(dataset_name=dataset, batch_size=2048, b=4096)
            # conf_8 = Config(dataset_name=dataset, batch_size=2048, b=8192, datasize=10)
            configs_b.append(conf_4)
            # configs_b.append(conf_8)
            for m in m_values:
                conf_q4 = Config(dataset_name=dataset, batch_size=2048, m=m, b=4096)
                # conf_q8 = Config(dataset_name=dataset, batch_size=2048, m=m, b=8192, datasize=10)
                configs_q.append(conf_q4)
                # configs_q.append(conf_q8)
        
        build_multiple_indexes_exp(EXP_NAME, configs_b)
        run_multiple_query_exp(EXP_NAME, configs_q)
        
    finally:
        heartbeat_process.terminate()