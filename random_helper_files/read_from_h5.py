import pandas as pd
import numpy as np

file = f"results/scratch_yandex100m/Yandex_100_r4_k2_m15_qps0.01_avg_rec0.538_bs=1024_reass=0_nr_ann=10_lr=0.001_chunk_size=5000_e=5_i=4.h5"

with pd.HDFStore(file, mode='r') as store:
    averages_df = store['averages']
    individual_df = store['individual_results']
    
    print(f"QPS: {averages_df['qps']}")
    print(f"Dist comps: {np.mean(individual_df['distance_computations'])}")