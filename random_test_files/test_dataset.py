from datasets import *

testdataobject = BigANNDataset(100)

fn = testdataobject.get_dataset_fn()

mmap = xbin_mmap(fn, testdataobject.dtype, maxn=testdataobject.nb)

queries = testdataobject.get_queries()

print(mmap.shape)
print(queries.shape)