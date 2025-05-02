import numpy as np
from sklearn.random_projection import SparseRandomProjection

import utils as ut
from config import Config
from bliss import load_indexes_and_models

def convert_to_sorted_random_projection(dataset, r, config: Config, index, model_directory, SIZE, target_dimensions):
    transformer = SparseRandomProjection(n_components=target_dimensions, random_state=42)
    mmp = np.memmap(f"models/{model_directory}/{config.dataset_name}_{config.datasize}_rp{target_dimensions}_r{r}.npy", mode ="w+", shape=(SIZE, config.rp_dim), dtype=np.float32)
    if SIZE > 10_000_000:
        start = 0
        for batch in dataset.get_dataset_iterator(bs=1_000_000):
            data = np.ascontiguousarray(batch, dtype=np.int32)
            reduced_vectors = transformer.fit_transform(data)
            end = len(batch) + start
            indices = index[start : end]
            mmp[start:end] = reduced_vectors[indices]
            start = end
    else:
        data = np.ascontiguousarray(dataset.get_dataset(), dtype=np.int32)
        reduced_vectors = transformer.fit_transform(data)
        indices = index[:]
        mmp[:] = reduced_vectors[indices]
    mmp.flush()

if __name__ == "__main__":
    # EDIT THIS CONFIG SO IT IS THE SAME AS THE MODEL YOU MADE
    config = Config("bigann", batch_size=1024, b=16384, reass_mode=2, reass_chunk_size=5000, rp_dim=8)
    dataset = ut.get_dataset_obj(config.dataset_name, size=config.datasize)
    dataset.prepare()
    SIZE = dataset.nb
    DIM = dataset.d
    b = config.b
    ((indexes, _), _) = load_indexes_and_models(config, SIZE, DIM, b)
    model_directory = f"{config.dataset_name}_r{config.r}_k{config.k}_b{config.b}_lr{config.lr}_bs={config.batch_size}_reass={config.reass_mode}_chunk_size={config.reass_chunk_size}_e={config.epochs}_i={config.iterations}"
    for r, index in enumerate(indexes, start=1):
        convert_to_sorted_random_projection(dataset, r, config, index, model_directory, SIZE, config.rp_dim)

