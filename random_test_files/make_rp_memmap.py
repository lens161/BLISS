import numpy as np
from sklearn.random_projection import SparseRandomProjection

import utils as ut
from config import Config
from bliss import load_indexes_and_models

# def convert_to_sorted_random_projection(data, config: Config, target_dimensions=8):
def convert_to_sorted_random_projection(data, r, config: Config, index, target_dimensions=8):
    relevant_data = np.ascontiguousarray(data, dtype=np.int32)
    transformer = SparseRandomProjection(n_components=target_dimensions, random_state=42)
    reduced_vectors = transformer.fit_transform(relevant_data)
    batch_size = 100_000
    N = len(data)
    # mmp = np.memmap(f"memmaps/{config.dataset_name}_{config.datasize}_rp{target_dimensions}.npy", mode ="w+", shape=reduced_vectors.shape, dtype=np.float32)
    mmp = np.memmap(f"memmaps/{config.dataset_name}_{config.datasize}_rp{target_dimensions}_r{r}.npy", mode ="w+", shape=reduced_vectors.shape, dtype=np.float32)
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        idx_batch = index[start:end]
        mmp[start:end] = reduced_vectors[idx_batch]
    mmp.flush()
    return reduced_vectors

if __name__ == "__main__":
    config = Config("glove-100-angular", batch_size=1000, b=4096)
    dataset = ut.get_dataset_obj(config.dataset_name, size=1)
    dataset.prepare()
    data = dataset.get_dataset()
    SIZE = dataset.nb
    DIM = dataset.d
    b = 4096
    ((indexes, _), _) = load_indexes_and_models(config, SIZE, DIM, b)
    for r, index in enumerate(indexes, start=1):
        convert_to_sorted_random_projection(data, r, config, index)
    # reduced_vectors = convert_to_sorted_random_projection(data, config)


#TODO:
# make 4 versions of the rp file, each sorted according to (inverted) index
# when getting candidates, we need to immediately get the rp vector and not just the id of the candidate (store as tuple)
# then when we do threshold filtering, we throw out the duplicates just like now
# if we filter out enough candidates with the rp ann, the access pattern for selecting the full candidates is still meh but we have way fewer candidates to select
