from config import Config
from bliss import run_bliss
from utils import get_best_device
import time 
import csv

def run_experiment(config: Config):
    start = time.time()
    recall = run_bliss(config)
    r = config.R
    k = config.K
    finish = time.time()

    elapsed = finish - start

    return (recall, r, k, elapsed)

def run_multiple_experiments(experiment_name, configs):
    results = []

    for config in configs:
        result = run_experiment(config)
        results.append(result)

    return experiment_name, results,

if __name__ == "__main__":
    configs = []

    range_K = 4

    dataset_name = "sift-128-euclidean"

    for i in range(1, range_K):
        config = Config(dataset_name, r= 1, k=i, epochs=2, iterations=2)
        configs.append(config)

    experiment_name, results = run_multiple_experiments("test_1", configs)

    recalls = []
    for result in results:
        recall = result[0]
        k = result[2]
        print(f"recall for k = {k} = {recall}")

    





    
