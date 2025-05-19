import numpy as np
from sklearn.random_projection import SparseRandomProjection

import utils as ut
from config import Config
from bliss import load_indexes_and_models

def convert_to_sorted_random_projection(dataset, r, config: Config, index, model_directory, SIZE):
    transformer = SparseRandomProjection(n_components=config.rp_dim, random_state=42)
    mmp = np.memmap(f"models/{model_directory}/{config.dataset_name}_{config.datasize}_rp{config.rp_dim}_r{r}.npy", mode ="w+", shape=(SIZE, config.rp_dim), dtype=np.float32)
    if SIZE > 10_000_000:
        start = 0
        reduced_vectors = np.empty((SIZE, config.rp_dim), dtype=np.float32)
        for batch in dataset.get_dataset_iterator(bs=1_000_000):
            end = len(batch) + start
            reduced_vectors[start : end] = transformer.fit_transform(batch)
            start = end
        indices = index[:]
        mmp[:] = reduced_vectors[indices]
    else:
        data = dataset.get_dataset()
        reduced_vectors = transformer.fit_transform(data)
        indices = index[:]
        mmp[:] = reduced_vectors[indices]
    mmp.flush()
    print(f"Finished rp memmap for r={r}")

if __name__ == "__main__":
    # EDIT THIS CONFIG SO IT IS THE SAME AS THE MODEL YOU MADE
    config = Config("Deep1B", batch_size=1024, b=16384, datasize=100, rp_dim=8)
    dataset = ut.get_dataset_obj(config.dataset_name, size=config.datasize)
    dataset.prepare()
    SIZE = dataset.nb
    DIM = dataset.d
    b = config.b
    ((indexes, _), _) = load_indexes_and_models(config, SIZE, DIM, b)
    model_directory = f"{config.dataset_name}_r{config.r}_k{config.k}_b{config.b}_lr{config.lr}_bs={config.batch_size}_reass={config.reass_mode}_chunk_size={config.reass_chunk_size}_e={config.epochs}_i={config.iterations}"
    for r, index in enumerate(indexes, start=1):
        convert_to_sorted_random_projection(dataset, r, config, index, model_directory, SIZE)