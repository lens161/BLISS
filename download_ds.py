from utils import get_dataset_obj

datasets = ["Deep1B", "Yandex", "MSSpaceV"]

for dataset in datasets:
    print(f"downloading {dataset}")
    ds = get_dataset_obj(dataset, 1000)
    ds.prepare()
    print(f"finished downloading {dataset}")

