import numpy as np

bucket_sizes = np.arange(10)

least_occupied = [2, 8, 6, 8]
least_occupied = np.array(least_occupied)
bucket_increments = np.bincount(least_occupied, minlength=len(bucket_sizes))

bucket_sizes = np.add(bucket_sizes, bucket_increments)
print(f"bucket_sizes {bucket_sizes}")
print(f"bucket increments {bucket_increments}")

