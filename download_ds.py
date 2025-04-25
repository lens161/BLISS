from utils import get_dataset_obj

ds = get_dataset_obj("bigann", 1000)

print(f"downloading bigann")
ds.prepare()
print(f"finished downloading bigann")

