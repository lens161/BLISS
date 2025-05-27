import gc
import logging
import os

import numpy as np

from config import Config
import utils as ut


def save_dataset_as_memmap(dataset, config: Config, SIZE, DIM):
    '''
    Put a dataset into a memmap and return the path where it was saved. 
    Small datasets can be loaded into memory and written to a memmap in one go, larger datasets are processed in chunks.
    '''
    logging.info("Creating dataset memmap")
    if not os.path.exists("memmaps/"):
        os.mkdir("memmaps/")
    memmap_path = f"memmaps/{config.dataset_name}_{config.datasize}.npy"
    mmp_shape = (SIZE, DIM)
    print(f"mmp shape = {mmp_shape}")
    mmp = None
    if not os.path.exists(memmap_path):
        mmp = np.memmap(memmap_path, mode ="w+", shape=mmp_shape, dtype=np.float32)
        if SIZE >= 10_000_000:
            fill_memmap_in_batches(dataset, mmp)
        else:
            data = dataset.get_dataset()[:]
            if dataset.distance() == "angular":
                data = ut.normalise_data(data)
            mmp[:] = data
            mmp.flush()
    del mmp
    return memmap_path, mmp_shape

def fill_memmap_in_batches(dataset, mmp):
    '''
    Save the dataset in a memmap in batches if the dataset is too large to load into memory in one go.
    '''
    index = 0
    for batch in dataset.get_dataset_iterator(bs=1_000_000):
        batch_size = len(batch)
        mmp[index: index + batch_size] = batch
        mmp.flush()
        del batch
        gc.collect()
        index += batch_size

config = Config("bigann", datasize=100, batch_size=1024, b=16384)
dataset_object = ut.get_dataset_obj(config.dataset_name, config.datasize)
dataset_object.prepare()
save_dataset_as_memmap(dataset_object, config, dataset_object.nb, dataset_object.d)