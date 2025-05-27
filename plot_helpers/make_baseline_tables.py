import ast
from collections import defaultdict
import glob
import matplotlib.pyplot as plt
import os
import pandas as pd

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

def convert_seconds_to_hms(seconds):
    """Convert seconds to hh:mm:ss format."""
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}"

def process_load_balance(load_balance_str):
    """Process the load balance column, extract last element of each sublist and compute the average."""
    try:
        # Parse the string representation of lists into actual list objects
        load_balance_list = ast.literal_eval(load_balance_str)
        
        # Extract the last element of each sublist
        last_values = [sublist[-1] for sublist in load_balance_list if isinstance(sublist, list)]
        
        # Compute the average of the last elements, return NaN if empty
        return sum(last_values) / len(last_values) if last_values else float('nan')
    except Exception as e:
        print(f"Error processing load balance: {e}")
        return float('nan')

def format_latex_table(results_by_dataset):
    table_rows = []

    for dataset, entries in results_by_dataset.items():
        # Sort entries by m
        entries = sorted(entries, key=lambda x: x['m'])

        # Collect per-m values in tabular format
        m_values = ' \\\\ '.join([f"{e['m']}" for e in entries])
        recall_values = ' \\\\ '.join([f"{e['recall']:.2f}" for e in entries])
        qps_values = ' \\\\ '.join([f"{e['qps']:.2f}" for e in entries])
        dist_comps_values = ' \\\\ '.join([f"{e['dist_comps']:.0f}" if pd.notna(e['dist_comps']) else 'x' for e in entries])

        # Per-dataset metrics: take from first entry (they're the same across m)
        first = entries[0]
        index_size = f"{first['index_size']:.1f}" if pd.notna(first['index_size']) else 'x'
        build_time = f"{first['build_time']}" if pd.notna(first['build_time']) else 'x'
        load_balance = f"{first['load_balance']:.2e}" if pd.notna(first['load_balance']) else 'x'

        row = (
            f"    {dataset} & "
            f"\\begin{{tabular}}[c]{{@{{}}r@{{}}}} {m_values} \\end{{tabular}} & "
            f"\\begin{{tabular}}[c]{{@{{}}l@{{}}}} {recall_values} \\end{{tabular}} & "
            f"\\begin{{tabular}}[c]{{@{{}}l@{{}}}} {qps_values} \\end{{tabular}} & "
            f"\\begin{{tabular}}[c]{{@{{}}l@{{}}}} {dist_comps_values} \\end{{tabular}} & "
            f"{index_size} & {build_time} & {load_balance} \\\\"
        )

        table_rows.append(row)

    # Final LaTeX table string
    table_latex = r"""\begin{table}[ht]
\centering
\begin{tabular}{l|l|lll|lll}
    \multicolumn{1}{c}{\textbf{Dataset}} & \multicolumn{1}{c}{\textbf{m}} & \multicolumn{1}{c}{\textbf{\begin{tabular}[c]{@{}c@{}}Recall\\ (10@10)\end{tabular}}} & \multicolumn{1}{c}{\textbf{QPS}} & \multicolumn{1}{c}{\textbf{\begin{tabular}[c]{@{}c@{}}\# Dist\\ comps\end{tabular}}} & \multicolumn{1}{c}{\textbf{\begin{tabular}[c]{@{}c@{}}Index\\ size\\ (MB)\end{tabular}}} & \multicolumn{1}{c}{\textbf{\begin{tabular}[c]{@{}c@{}}Build\\ time\\ (hh:mm:ss) \end{tabular}}} & \multicolumn{1}{c}{\textbf{\begin{tabular}[c]{@{}c@{}}Load\\ balance\end{tabular}}} \\
\hline
""" + '\n\\hline\n'.join(table_rows) + r"""
\end{tabular}
\caption{Baseline metrics for X.}
\label{tab:X_baseline}
\end{table}
"""
    return table_latex

def make_baseline_stats_table(experiment_name, build_csv_dir="."):
    query_files = find_query_files(experiment_name)
    query_results, query_averages = compile_query_results(query_files)
    output_path = f"results/{experiment_name}/baseline_table.tex"
    # Step 1: Load build info CSVs, one per dataset (from the same directory as the .hdf5 files)
    build_info = {}

    # Get the directory of the first .hdf5 file to use for finding build files
    if query_files:
        build_csv_dir = os.path.dirname(query_files[0])  # Folder containing the .hdf5 files
    else:
        raise FileNotFoundError("No query files found.")

    # Debugging: Check the build files found by glob
    build_files = glob.glob(os.path.join(build_csv_dir, "*_build.csv"))
    print(f"Found build files: {build_files}")  # Debugging line to check the files
    
    # Load build files and store info
    for csv_file in build_files:
        df = pd.read_csv(csv_file)

        # Extract dataset name from the filename (assumes the dataset name is the part before the first '_')
        dataset_name = os.path.basename(csv_file).split('_')[0]

        # Compute index size by summing 'index_sizes_total' and 'model_sizes_total' columns
        index_size = df['index_sizes_total'].iloc[0] + df['model_sizes_total'].iloc[0]  # Assuming these are numeric
        build_time = df['build_time'].iloc[0]  # Assuming this exists

        # Process load balance (it should be a list of lists in string format)
        load_balance_str = df['load_balances'].iloc[0]  # Assuming this is the column to parse
        load_balance = process_load_balance(load_balance_str)

        # Store the data for each dataset
        build_info[dataset_name] = {
            'index_size': index_size,
            'build_time': build_time,
            'load_balance': load_balance
        }
    print(build_info)

    # Step 2: Aggregate query results
    results_by_dataset = defaultdict(list)

    for i, result in enumerate(query_results):
        avg_df = query_averages[i]
        result_df = query_results[i]  # This contains individual query results

        for _, row in avg_df.iterrows():
            dataset_name = row['dataset_name']
            m = int(row['m'])
            recall = float(row['avg_recall'])
            qps = float(row['qps'])

            # Compute average number of distance comparisons from the individual results
            dist_comps = result_df['distance_computations'].mean() if 'distance_computations' in result_df.columns else float('nan')

            # Load per-dataset build info (computed in Step 1)
            if dataset_name in build_info:
                build_data = build_info[dataset_name]
                index_size = build_data['index_size']
                build_time = build_data['build_time']
                load_balance = build_data['load_balance']
            else:
                print(f"Warning: Build info missing for dataset '{dataset_name}', using 'x'.")
                index_size = build_time = load_balance = float('nan')

            # Convert build time to hh:mm:ss format
            if pd.notna(build_time):
                build_time = convert_seconds_to_hms(build_time)

            # Create an entry for this dataset and m combination
            entry = {
                'm': m,
                'recall': recall,
                'qps': qps,
                'dist_comps': dist_comps,
                'index_size': index_size,
                'build_time': build_time,
                'load_balance': load_balance,
            }

            results_by_dataset[dataset_name].append(entry)

    # Step 3: Format LaTeX rows
    latex_table = format_latex_table(results_by_dataset)

    # Step 4: Output LaTeX to file
    with open(output_path, "w") as f:
        f.write(latex_table)
    print(f"LaTeX table written to {output_path}")

if __name__ == "__main__":
    make_baseline_stats_table("1m_baseline_stats")