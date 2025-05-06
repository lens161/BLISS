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
    # filepaths = [f for f in filepaths if not f.endswith('build.csv') and not f.endswith('memory_log.csv')] # filter out non-query files: temporarily disabled
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
    memory = float(averages[i]['memory'].iloc[0])
    return dataset_name, datasize, r, k, m, qps, avg_recall, bs, reass_mode, nr_ann, lr, chunk_size, epochs, iters, memory

def plot_individual_recall_vs_dist_comps(results, averages, experiment_name):
    '''
    Make a plot where recall is compared to nr of distance computations (nr of candidates when candidate set was too large and required true distance computations for reordering).
    '''
    for i, result in enumerate(results):
        dataset_name, datasize, r, k, m, qps, avg_recall, bs, reass_mode, nr_ann, lr, chunk_size, epochs, iters, memory = get_parameters_and_stats(averages, i)

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
        
        # reuse qps from filename and make plot
        plt.savefig(f"results/{experiment_name}/{dataset_name}_{datasize}_r{r}_k{k}_m{m}_qps{qps:.2f}_avg_rec{avg_recall:.3f}_bs={bs}_reass={reass_mode}_nr_ann={nr_ann}_lr={lr}_chunk_size={chunk_size}_e={epochs}_i={iters}.png", dpi=300)
        
        # alternatively, calculate recall and qps from individual queries, but qps measurement is slightly off
        # new_recall = result['recall'].mean()
        # new_qps = len(result) / result['elapsed'].sum()
        # plt.savefig(f"results/{experiment_name}/r{r}_k{k}_m{m}_qps{new_qps:.2f}_avg_rec{avg_recall:.3f}_bs={bs}_reass={reass_mode}_nr_ann={nr_ann}_lr={lr}.png", dpi=300)

def plot_recall_vs_dist_comps_per_m(results, averages, experiment_name):
    stats = []
    for i, result in enumerate(results):
        avg_recall = averages[i]['avg_recall']
        avg_dist_comps = result['distance_computations'].mean()
        m = int(averages[i]['m'].iloc[0])
        stats.append({'m': m, 'avg_recall': avg_recall, 'avg_dist_comps': avg_dist_comps})
    
    stats.sort(key=lambda x: x['avg_dist_comps'])
    x = [result['avg_dist_comps'] for result in stats]
    y = [result['avg_recall'] for result in stats]
    plt.figure(figsize=(8, 5))
    plt.plot(x, y, marker='x', linestyle='-', color='b', label='Recall vs Distance Computations')

    # Labels and title
    plt.xlabel('Number of Distance Computations')
    plt.ylabel('Recall')
    plt.title('Recall vs. Distance Computations')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"results/{experiment_name}/recall_vs_dist_comps_per_m.png", dpi=300)

def plot_build_time_vs_chunk_size(experiment_name):
    """
    Read a build-log CSV and plot chunk size vs total build time,
    with separate lines for each reass_mode found in the data.
    """
    csv_file = f"results/{experiment_name}/{experiment_name}_build.csv"
    df = pd.read_csv(csv_file)

    out_dir = f"results/{experiment_name}"
    os.makedirs(out_dir, exist_ok=True)

    plt.figure(figsize=(8, 5))
    for mode in sorted(df['reass_mode'].unique()):
        subset = df[df['reass_mode'] == mode]
        # sort by chunk size for a clean line
        subset = subset.sort_values('reass_chunk_size')
        plt.plot(
            subset['reass_chunk_size'],
            subset['build_time'],
            marker='o',
            linestyle='-',
            label=f"Mark {mode}"
        )

    plt.xlabel("Chunk Size")
    plt.ylabel("Total Build Time (s)")
    plt.title(f"Build Time vs Chunk Size ({experiment_name})")
    plt.grid(True)
    plt.legend(title="Reassign Mode")
    plt.tight_layout()

    plt.savefig(f"{out_dir}/build_time_vs_chunk_size_by_mode.png", dpi=300)
    plt.close()

# def plot_recall_vs_dist_comps_per_m_per_dataset(results, averages, experiment_name):
#     # Group data by dataset_name
#     dataset_groups = {}
#     for i, result in enumerate(results):
#         dataset_name = averages[i]['dataset_name']
#         avg_recall = averages[i]['avg_recall']
#         avg_dist_comps = result['distance_computations'].mean()
#         m = int(averages[i]['m'].iloc[0])
        
#         if dataset_name not in dataset_groups:
#             dataset_groups[dataset_name] = []
#         dataset_groups[dataset_name].append({'m': m, 'avg_recall': avg_recall, 'avg_dist_comps': avg_dist_comps})
    
#     # Define a list of colors for different datasets
#     colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']  # Modify as needed if there are more datasets

#     plt.figure(figsize=(8, 5))
    
#     # Iterate over the dataset groups and plot each one
#     for idx, (dataset_name, group) in enumerate(dataset_groups.items()):
#         group.sort(key=lambda x: x['avg_dist_comps'])
#         x = [item['avg_dist_comps'] for item in group]
#         y = [item['avg_recall'] for item in group]
        
#         # Use the color from the list or loop back through the colors
#         color = colors[idx % len(colors)]
        
#         # Plot each dataset with its color and a specific marker
#         plt.plot(x, y, marker='o', linestyle='-', color=color, label=dataset_name)
    
#     # Labels and title
#     plt.xlabel('Number of Distance Computations')
#     plt.ylabel('Recall')
#     plt.title('Recall vs. Distance Computations')
#     plt.legend(title="Dataset Name")
#     plt.grid(True)
#     plt.tight_layout()

#     # Save the plot to a file
#     plt.savefig(f"results/{experiment_name}/recall_vs_dist_comps_per_m.png", dpi=300)


def make_plots(experiment_name):
    '''
    Include all plot functions that should be run here.
    '''
    # get results from query files
    # query_files = find_query_files(experiment_name)
    # query_results, query_averages = compile_query_results(query_files)
    plot_build_time_vs_chunk_size(experiment_name)
    # TODO: get results from build and memory files

    # make plots for whole experiment, add more plot functions as needed
    plot_individual_recall_vs_dist_comps(query_results, query_averages, experiment_name)
    plot_recall_vs_dist_comps_per_m(query_results, query_averages, experiment_name)
    # plot_recall_vs_dist_comps_per_m_per_dataset(query_results, query_averages, experiment_name)

if __name__ == "__main__":
    make_plots("test_twostep_deep1B_baseline")