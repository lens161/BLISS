from config import Config
from bliss import run_bliss
from utils import get_best_device
import numpy as np
import time 
import csv
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
        r = config.R
        k = config.K
        epochs = config.EPOCHS
        iterations = config.ITERATIONS
        time_per_r, build_time, memory_usage = run_bliss(config, mode=mode)
        stats.append({'R':r, 'k':k, 'epochs_per_it':epochs, 'iterations':iterations, 'build_time':build_time, 'mem':memory_usage})
        print(time_per_r)
    df = pd.DataFrame(stats)
    df.to_csv(f"{experiment_name}.csv")

def run_multiple_query_exp(experiment_name, configs):
    mode = 'query'
    for config in configs:
        r = config.R
        k = config.K
        m =config.M
        results = []
        total_query_time, stats, avg_recall  = run_bliss(config, mode=mode)
        for (anns, dist_comps, elapsed, recall) in stats:
            results.append({'ANNs': anns, 
                            'distance_computations': dist_comps, 
                            'elapsed': elapsed,
                            'recall': recall})
        qps = len(stats)/total_query_time
        df = pd.DataFrame(results)
        df.to_csv(f"{experiment_name}_r{r}_k{k}_m{m}_qps{qps:.2f}_avg_rec{avg_recall:.3f}.csv", index=False)
        plt.figure(figsize=(8, 5))
        plt.scatter(df['distance_computations'], df['recall'], color='blue', s=20)
        plt.xlabel("Distance Computations")
        plt.ylabel("Recall")
        plt.title(f"Distance Computations vs Recall R={r} k={k} m={m}")
        plt.grid(True)
        plt.savefig(f"{experiment_name}_r{r}_k{k}_m{m}_qps{qps:.2f}_avg_rec{avg_recall:.3f}.png", dpi=300)

    return experiment_name, avg_recall, total_query_time, results

if __name__ == "__main__":
    configs_q = []
    configs_b = []

    range_M = 5
    range_K = 5
    range_threshold = 5

    dataset_name = "sift-128-euclidean"

    for i in range(1, range_M):
        config = Config(dataset_name, r= 4, k=2, m=i, epochs=1, iterations=1)
        configs_q.append(config)

    experiment_name, avg_recall, time_all_queries, stats = run_multiple_query_exp("test_1", configs_q)

    for i in range(1, range_K):
        config = Config(dataset_name, k=i)
        configs_b.append(config)


    # # code below was used to plot single graph from an existing csv
    # df = pd.DataFrame(pd.read_csv("test_1_r4_k2_m3_qps5.47_avg_rec0.973.csv"))

    # num_bins = 10
    # bins = np.linspace(df['distance_computations'].min(), df['distance_computations'].max(), num_bins+1)

    # # Bin the data
    # df['bin'] = pd.cut(df['distance_computations'], bins=bins)

    # # Group by the bin and calculate the average recall for each bin.
    # grouped = df.groupby('bin')['recall'].mean()

    # # Compute the midpoints of each bin for the x-axis
    # bin_midpoints = [interval.mid for interval in grouped.index.categories]

    # # Plot a line graph
    # plt.figure(figsize=(8, 5))
    # plt.plot(bin_midpoints, grouped.values, marker='o', linestyle='-')
    # plt.xlabel("Distance Computations (binned midpoint)")
    # plt.ylabel("Average Recall")
    # plt.title("Binned Average Recall vs Distance Computations")
    # plt.grid(True)
    # plt.savefig("test2.png", dpi = 300)
    # plt.show()

    # # df = pd.DataFrame(stats)
    # # df.to_csv(f"{experiment_name}_avg_rec:{avg_recall:.3f}.csv", index = False)
