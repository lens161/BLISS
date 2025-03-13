import ast
import glob
import csv
import os
import matplotlib.pyplot as plt
import pandas as pd

def find_files(experiment_name):
    parameters_per_file = []
    filepaths = glob.glob(f"results/{experiment_name}/*.csv")
    filepaths = [f for f in filepaths if not f.endswith('build.csv')]
    for filepath in filepaths:
        filename = filepath.split("/")[-1]
        filename_components = filename.split("_")
        parameters = {'r': filename_components[0], 'k': filename_components[1], 'm': filename_components[2]}
        parameters_per_file.append(parameters)
    return filepaths, parameters_per_file

def compile_results(files):
    results = []
    dtype_dict = {
        'elapsed': float,
        'recall': float,
        'distance_computations': float,
        'ANNs': str
    }
    for file in files:
        df = pd.read_csv(file, dtype=dtype_dict, sep=r'\s*,\s*', engine='python')
        results.append(df)
    return results

def make_recall_vs_distance_comps_plots(results, parameters_per_file, experiment_name):
    for i, result in enumerate(results):
        r = parameters_per_file[i]['r']
        k = parameters_per_file[i]['k']
        m = parameters_per_file[i]['m']
        plt.figure(figsize=(8, 5))
        plt.scatter(result['distance_computations'], result['recall'], color='blue', s=20)
        plt.xlabel("Distance Computations")
        plt.ylabel("Recall")
        plt.title(f"Distance Computations vs Recall R={r} k={k} m={m}")
        plt.grid(True)
        foldername = f"results/{experiment_name}"
        if not os.path.exists("results"):
            os.mkdir("results")
        if not os.path.exists(f"results/{experiment_name}"):
            os.mkdir(foldername)
        qps = len(result) / result['elapsed'].sum()
        avg_recall = result['recall'].mean()
        plt.savefig(f"results/{experiment_name}/{r}_{k}_{m}_qps{qps:.2f}_avg_rec{avg_recall:.3f}.png", dpi=300)

def make_plots(results, parameters_per_file, experiment_name):
    make_recall_vs_distance_comps_plots(results, parameters_per_file, experiment_name)

if __name__ == "__main__":
    experiment_name = "plot_test"
    files, parameters_per_file = find_files(experiment_name)
    results = compile_results(files)
    make_plots(results, parameters_per_file, experiment_name)