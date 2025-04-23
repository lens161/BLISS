import numpy as np
from sklearn.random_projection import SparseRandomProjection

import utils as ut
from config import Config
from bliss import load_indexes_and_models


def convert_to_sorted_random_projection(data, config: Config, target_dimensions=8):
    relevant_data = np.ascontiguousarray(data, dtype=np.int32)
    transformer = SparseRandomProjection(n_components=target_dimensions, random_state=42)
    reduced_vectors = transformer.fit_transform(relevant_data)
    mmp = np.memmap(f"memmaps/{config.dataset_name}_{config.datasize}_rp{target_dimensions}.npy", mode ="w+", shape=reduced_vectors.shape, dtype=np.float32)
    mmp[:] = reduced_vectors
    mmp.flush()
    return reduced_vectors

if __name__ == "__main__":
    config = Config("sift-128-euclidean", batch_size=2048, b=4096)
    dataset = ut.get_dataset_obj(config.dataset_name, size=1)
    dataset.prepare()
    data = dataset.get_dataset()
    # ((indexes, _), _) = load_indexes_and_models(config)
    # for r, index in enumerate(indexes, start=1):
    #     convert_to_sorted_random_projection(data, r, config)
    reduced_vectors = convert_to_sorted_random_projection(data, config)
    memmap_rp_path = f"memmaps/{config.dataset_name}_{config.datasize}_rp8.npy"
    SIZE = 1_000_000