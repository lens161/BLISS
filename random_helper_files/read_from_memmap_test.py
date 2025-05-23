import utils as ut
import numpy as np
import time
from config import Config

config = Config("bigann", datasize=100, b=16384, batch_size=1024)
dataset = ut.get_dataset_obj(config.dataset_name, config.datasize)
dataset.prepare()

mmap = dataset.get_dataset_memmap()

nr_datapoints = 1000
random_indices = np.random.randint(dataset.nb, size=nr_datapoints)

start = time.time()
mmap[random_indices]
end = time.time()
print(f"Took {end-start} to take {nr_datapoints} items")