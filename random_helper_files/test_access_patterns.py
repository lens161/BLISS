import time
import numpy as np
import utils as ut
from config import Config
from bliss import load_indexes_and_models

if __name__ == "__main__":
    config = Config("sift-128-euclidean", batch_size=2048, b=4096)
    dataset = ut.get_dataset_obj(config.dataset_name, size=1)
    dataset.prepare()
    data = dataset.get_dataset()
    spread_pattern_sum = 0
    connected_pattern_sum = 0

    chunk_size = 10000
    
    for i in range(0,10000):
        random_indices = np.random.randint(len(data), size=chunk_size)
        random_start_index = np.random.randint((len(data)-chunk_size), size=1)[0]

        spread_pattern_s = time.time()
        selected = data[random_indices]
        spread_pattern_e = time.time()
        spread_pattern_sum += (spread_pattern_e-spread_pattern_s)

        connected_pattern_s = time.time()
        selected2 = data[random_start_index:random_start_index+chunk_size]
        connected_pattern_e = time.time()
        connected_pattern_sum += (connected_pattern_e-connected_pattern_s)

    print(f"Time for random pattern: {spread_pattern_sum:.10f}, Connected pattern: {connected_pattern_sum:.10f}")