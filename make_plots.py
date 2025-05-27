import glob
import matplotlib.pyplot as plt
import os
import pandas as pd


def find_build_file(experiment_name):
    filepath = glob.glob(f"results/{experiment_name}/*build.csv")[0] # list of only 1 file
    return filepath

def find_query_files(experiment_name):
    '''
    Find all the .csv files in an experiment folder to collect the results for different inference parameter combinations within an experiment.
    Build and memory log files are excluded.
    '''
    filepaths = glob.glob(f"results/{experiment_name}/*.h5") # find all .h5 files in folder
    return filepaths

def compile_query_results(files):
    '''
    Collect results from a set of csv files in one dataframe.
    '''
    all_results = []
    all_averages = []
    for file in files:
        with pd.HDFStore(file, mode='r') as store:
            averages = store['averages']
            individual_results = store['individual_results']
            all_averages.append(averages)
            all_results.append(individual_results)
    return all_results, all_averages

def get_parameters_and_stats(averages, i):
    dataset_name = averages[i]['dataset_name'].iloc[0]
    datasize = int(averages[i]['datasize'].iloc[0])
    r = int(averages[i]['r'].iloc[0])
    k = int(averages[i]['k'].iloc[0])
    m = int(averages[i]['m'].iloc[0])
    qps = float(averages[i]['qps'].iloc[0])
    avg_recall = float(averages[i]['avg_recall'].iloc[0])
    bs = int(averages[i]['bs'].iloc[0])
    reass_mode = int(averages[i]['reass_mode'].iloc[0])
    nr_ann = int(averages[i]['nr_ann'].iloc[0])
    lr = float(averages[i]['lr'].iloc[0])
    chunk_size = int(averages[i]['chunk_size'].iloc[0])
    epochs = int(averages[i]['e'].iloc[0])
    iters = int(averages[i]['i'].iloc[0])
    query_twostep = bool(averages[i]['query_twostep'].iloc[0])
    twostep_limit = int(averages[i]['twostep_limit'].iloc[0])
    return dataset_name, datasize, r, k, m, qps, avg_recall, bs, reass_mode, nr_ann, lr, chunk_size, epochs, iters, query_twostep, twostep_limit

def plot_individual_recall_vs_dist_comps(results, averages, experiment_name):
    '''
    Make a plot where recall is compared to nr of distance computations (nr of candidates when candidate set was too large and required true distance computations for reordering).
    '''
    for i, result in enumerate(results):
        dataset_name, datasize, r, k, m, qps, avg_recall, bs, reass_mode, nr_ann, lr, chunk_size, epochs, iters, query_twostep, twostep_limit = get_parameters_and_stats(averages, i)

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
        
        plt.savefig(f"results/{experiment_name}/{dataset_name}_{datasize}_r{r}_k{k}_m{m}_qps{qps:.2f}_avg_rec{avg_recall:.3f}_bs={bs}_reass={reass_mode}_nr_ann={nr_ann}_lr={lr}_chunk_size={chunk_size}_e={epochs}_i={iters}_twostep={query_twostep}_limit={twostep_limit}.png", dpi=300)
        
        plt.close()

def plot_recall_vs_dist_comps_per_m_per_dataset(results, averages, experiment_name):

    # Mapping of full dataset names to custom legend labels
    custom_labels = {
        'sift-128-euclidean': 'SIFT',
        'glove-100-angular': 'GloVe',
        # Add more mappings as needed
    }

    # Group data by dataset_name
    dataset_groups = {}
    for i, result in enumerate(results):
        dataset_name = averages[i]['dataset_name'].iloc[0]  # Get actual string
        avg_recall = averages[i]['avg_recall']
        avg_dist_comps = result['distance_computations'].mean()
        m = int(averages[i]['m'].iloc[0])
        
        if dataset_name not in dataset_groups:
            dataset_groups[dataset_name] = []
        dataset_groups[dataset_name].append({'m': m, 'avg_recall': avg_recall, 'avg_dist_comps': avg_dist_comps})
    
    colors = ['b', 'r', 'g', 'c', 'm', 'y', 'k']

    plt.figure(figsize=(8, 3))

    for idx, (dataset_name, group) in enumerate(dataset_groups.items()):
        group.sort(key=lambda x: x['avg_dist_comps'])
        x = [item['avg_dist_comps'] for item in group]
        y = [item['avg_recall'] for item in group]

        color = colors[idx % len(colors)]
        label = custom_labels.get(dataset_name, dataset_name)  # Default to full name if not mapped
        
        plt.plot(x, y, marker='x', linestyle='-', color=color, label=label)

    plt.xlabel('Number of distance computations')
    plt.ylabel('Recall')
    plt.title('Recall vs. distance computations')
    plt.legend(title="Dataset", loc='lower right')
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(f"results/{experiment_name}/recall_vs_dist_comps_per_m.svg", dpi=300)


def make_plots(experiment_name):
    '''
    Include all plot functions that should be run here.
    '''
    # get results from query files
    query_files = find_query_files(experiment_name)
    query_results, query_averages = compile_query_results(query_files)

    # # make plots for whole experiment, add more plot functions as needed
    plot_individual_recall_vs_dist_comps(query_results, query_averages, experiment_name)
    plot_recall_vs_dist_comps_per_m_per_dataset(query_results, query_averages, experiment_name)

if __name__ == "__main__":
    make_plots("1m_combined_diff_ms")