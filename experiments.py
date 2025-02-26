from config import Config
from bliss import run_bliss
from utils import get_best_device
import time 
import csv
import pandas as pd

def run_experiment(config: Config, mode = 'query'):
    # TO-DO: 
    # - get statistics per query
    # - seperate query statistics from building statistics
    avg_recall, stats, total_query_time = run_bliss(config, mode= mode)
    return total_query_time, avg_recall, stats

def build_multiple_indexes_exp(experiment_name, configs):
    mode = 'build'
    for config in configs:
        time_per_r = run_bliss(config, mode)
        print(time_per_r)
        
def run_multiple_query_exp(experiment_name, configs):
    mode = 'query'
    for config in configs:
        r = config.R
        k = config.K
        m =config.M
        results = []
        total_query_time, avg_recall, stats = run_experiment(config, mode=mode)
        for (anns, dist_comps, elapsed, recall) in stats:
            results.append({'ANNs': anns, 
                            'distance computations': dist_comps, 
                            'elapsed': elapsed,
                            'recall': recall})
        qps = len(stats)/total_query_time
        df = pd.DataFrame(results)
        df.to_csv(f"{experiment_name}_r{r}_k{k}_m{m}_qps{qps:.2f}_avg_rec{avg_recall:.3f}.csv", index=False)

    return experiment_name, avg_recall, total_query_time, results

if __name__ == "__main__":
    configs = []

    range_M = 5

    dataset_name = "sift-128-euclidean"

    for i in range(1, range_M):
        config = Config(dataset_name, r= 1, k=2, m=i, epochs=1, iterations=1)
        configs.append(config)

    experiment_name, avg_recall, time_all_queries, stats = run_multiple_query_exp("test_1", configs)

    # df = pd.DataFrame(stats)
    # df.to_csv(f"{experiment_name}_avg_rec:{avg_recall:.3f}.csv", index = False)
