import glob
import matplotlib.pyplot as plt
import os
import pandas as pd


def find_files(experiment_name):
    '''
    Find all the .csv files in an experiment folder to collect the results for different inference parameter combinations within an experiment.
    Currently, build file is excluded as it is describing results of index building.
    '''
    parameters_per_file = []
    filepaths = glob.glob(f"results/{experiment_name}/*.csv")
    filepaths = [f for f in filepaths if not f.endswith('build.csv') and not f.endswith('memory_log.csv')]
    for filepath in filepaths:
        filename = filepath.split("/")[-1][:-4]
        filename_components = filename.split("_")
        print(filename_components)
        parameters = {'r': filename_components[0], 'k': filename_components[1], 'm': filename_components[2], 'qps': filename_components[3], 'rec': filename_components[5], 'bs': filename_components[6], 'reass_mode': filename_components[7], 'nr_ann': "nr_"+filename_components[9], 'lr': filename_components[10]}
        parameters_per_file.append(parameters)
    return filepaths, parameters_per_file

def compile_results(files):
    '''
    Collect results from a set of csv files in one dataframe.
    '''
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
    '''
    Make a plot where recall is compared to nr of distance computations (nr of candidates when candidate set was too large and required true distance computations for reordering).
    '''
    for i, result in enumerate(results):
        r = parameters_per_file[i]['r']
        k = parameters_per_file[i]['k']
        m = parameters_per_file[i]['m']
        qps = parameters_per_file[i]['qps']
        # rec = parameters_per_file[i]['rec']
        bs = parameters_per_file[i]['bs']
        reass_mode = parameters_per_file[i]['reass_mode']
        nr_ann = parameters_per_file[i]['nr_ann']
        lr = parameters_per_file[i]['lr']

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
        
        avg_recall = result['recall'].mean()
        plt.savefig(f"results/{experiment_name}/{r}_{k}_{m}_{qps}_avg_rec{avg_recall:.3f}_{bs}_{reass_mode}_{nr_ann}_{lr}.png", dpi=300)
        # new_qps = len(result) / result['elapsed'].sum()
        # plt.savefig(f"results/{experiment_name}/{r}_{k}_{m}_qps{new_qps:.2f}_avg_rec{avg_recall:.3f}_{bs}_{reass_mode}_{nr_ann}_{lr}.png", dpi=300)

def make_plots(results, parameters_per_file, experiment_name):
    '''
    Include all plot functions that should be run here.
    '''
    make_recall_vs_distance_comps_plots(results, parameters_per_file, experiment_name)

if __name__ == "__main__":
    experiment_name = "check_qps_plots"
    files, parameters_per_file = find_files(experiment_name)
    results = compile_results(files)
    make_plots(results, parameters_per_file, experiment_name)