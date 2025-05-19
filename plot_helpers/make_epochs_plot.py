import ast
import glob
import os
import matplotlib
from matplotlib.ticker import MultipleLocator
import pandas as pd
import matplotlib.pyplot as plt

matplotlib.rcParams.update({
    "font.size": 16,
    "svg.fonttype": "none",
})

def process_load_balance(load_balance_str):
    """Parse and average the last values of the nested list."""
    try:
        load_balance_list = ast.literal_eval(load_balance_str)
        last_values = [sublist[-1] for sublist in load_balance_list if isinstance(sublist, list)]
        return sum(last_values) / len(last_values) if last_values else float('nan')
    except Exception as e:
        print(f"Error parsing load balance: {e}")
        return float('nan')

def collect_data_for_plot(hdf5_dir, m, include_qps=False):
    """
    Extract:
    - epoch count = (e + 1) * (i + 1)
    - recall or QPS (depending on include_qps flag)
    - load balance, matched by matching parameters (epochs, iterations) between h5 file and build csv
    """
    data = []

    # Load the build.csv file
    build_files = glob.glob(os.path.join(hdf5_dir, "*_build.csv"))
    if not build_files:
        raise FileNotFoundError("No build.csv file found.")
    build_df = pd.read_csv(build_files[0])

    # Find HDF5 files in the same directory
    hdf5_files = glob.glob(os.path.join(hdf5_dir, "*.h5"))
    
    for h5_file in hdf5_files:
        try:
            with pd.HDFStore(h5_file, mode='r') as store:
                averages_df = store['averages']

                # Filter for m
                filtered_df = averages_df[averages_df['m'] == m]

                if filtered_df.empty:
                    continue

                row = filtered_df.iloc[0]
                e = int(row['e'])  # HDF5 epoch count
                i = int(row['i'])  # HDF5 iteration count
                epoch_count = e * i

                # Extract recall or QPS
                if include_qps:
                    qps = float(row['qps'])
                    query_time_ms = 1000 / qps if qps > 0 else float('nan')
                    recall = None
                else:
                    qps = None
                    query_time_ms = None
                    recall = float(row['avg_recall'])

                # Match the parameters from averages with build.csv (match on epochs and iterations)
                match = build_df[
                    (build_df['epochs_per_it'] == e) & 
                    (build_df['iterations'] == i)
                ]
                
                if match.empty:
                    print(f"Warning: No matching row in build.csv for epochs={e}, iterations={i} in {h5_file}")
                    continue
                elif len(match) > 1:
                    print(f"Warning: Multiple matches found in build.csv for epochs={e}, iterations={i} in {h5_file}")
                    continue

                load_balance_str = match.iloc[0]['load_balances']
                load_balance = process_load_balance(load_balance_str)

            # Append the results
            data.append((epoch_count, recall, qps, query_time_ms, load_balance))

        except Exception as e:
            print(f"Error processing {h5_file}: {e}")

    return pd.DataFrame(data, columns=['epoch_count', 'recall', 'qps', 'query_time_ms', 'load_balance'])

def plot_load_balance_and_recall(df, directory, m):
    """Plot load balance and recall against epoch count with dual Y axes."""
    df = df.sort_values(by='epoch_count')

    fig, ax1 = plt.subplots(figsize=(7, 6))

    ax1.tick_params(axis='x', labelsize=14)
    ax1.xaxis.set_major_locator(MultipleLocator(5))

    color1 = 'tab:blue'
    ax1.set_xlabel('Epochs', fontsize=10, labelpad=15)
    ax1.set_ylabel('Load Balance', color=color1, fontsize=10)
    line1, = ax1.plot(df['epoch_count'], df['load_balance'], color=color1, marker='x', linestyle='--',
                      label='Load Balance', linewidth=3, markersize=8, markeredgewidth=2)
    ax1.tick_params(axis='y', labelcolor=color1, labelsize=8)
    ax1.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

    ax2 = ax1.twinx()
    color2 = 'tab:red'
    ax2.set_ylabel(f'Recall (m={m})', color=color2, fontsize=10)
    line2, = ax2.plot(df['epoch_count'], df['recall'], color=color2, marker='o', label=f'Recall (m={m})',
                      linewidth=3, markersize=8, markeredgewidth=2)
    ax2.tick_params(axis='y', labelcolor=color2, labelsize=8)

    lines = [line1, line2]
    labels = [line.get_label() for line in lines]

    ax2.legend(
        lines,
        labels,
        loc='center right',
        bbox_to_anchor=(1.00, 0.5),
        fontsize=8,
        borderpad=0.3,
        borderaxespad=1,
        framealpha=0.8,
        edgecolor='black',
        handlelength=2,
        handleheight=2,
    )

    fig.suptitle(f'Load Balance and Recall (m={m}) vs Epochs', fontsize=12)
    fig.tight_layout()
    plt.grid(True)

    output_path = os.path.join(directory, "load_balance_and_recall_per_epochs.svg")
    plt.savefig(output_path, format='svg', bbox_inches='tight')
    plt.close()
    print(f"Plot saved to: {output_path}")

def plot_load_balance_and_query_time(df, directory, m):
    """Plot load balance and query time against epoch count with dual Y axes."""
    df = df.sort_values(by='epoch_count')

    fig, ax1 = plt.subplots(figsize=(11, 9))

    ax1.tick_params(axis='x', labelsize=14)
    ax1.xaxis.set_major_locator(MultipleLocator(5))

    color1 = 'tab:blue'
    ax1.set_xlabel('Epochs', fontsize=18, labelpad=15)
    ax1.set_ylabel('Load Balance', color=color1, fontsize=18, labelpad=15)
    line1, = ax1.plot(df['epoch_count'], df['load_balance'], color=color1, marker='x', linestyle='--',
                      label='Load Balance', linewidth=3, markersize=8, markeredgewidth=2)
    ax1.tick_params(axis='y', labelcolor=color1, labelsize=14)
    ax1.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

    ax2 = ax1.twinx()
    color2 = 'tab:red'
    ax2.set_ylabel('Query Time (ms/query)', color=color2, fontsize=18, labelpad=15)
    line2, = ax2.plot(df['epoch_count'], df['query_time_ms'], color=color2, marker='o', label='Query Time',
                      linewidth=3, markersize=8, markeredgewidth=2)
    ax2.tick_params(axis='y', labelcolor=color2, labelsize=14)

    lines = [line1, line2]
    labels = [line.get_label() for line in lines]

    ax2.legend(
        lines,
        labels,
        loc='center right',
        bbox_to_anchor=(1.00, 0.5),
        fontsize=12,
        title_fontsize=12,
        borderpad=0.3,
        borderaxespad=1,
        framealpha=0.8,
        edgecolor='black',
        handlelength=2,
        handleheight=2,
    )

    fig.suptitle(f'Load Balance and Query Time vs Epochs', fontsize=20)
    fig.tight_layout()
    plt.grid(True)

    output_path = os.path.join(directory, "load_balance_and_query_time_per_epochs.svg")
    plt.savefig(output_path, format='svg', bbox_inches='tight')
    plt.close()
    print(f"Plot saved to: {output_path}")

# Main usage
if __name__ == "__main__":
    directory = "results/glove_diff_epochs_v5_nomem"  # Adjust as needed
    m = 15

    # Collect data for both plots
    df_recall = collect_data_for_plot(directory, m, include_qps=False)
    df_qtime = collect_data_for_plot(directory, m, include_qps=True)

    # Plot for Recall
    plot_load_balance_and_recall(df_recall, directory, m)

    # Plot for Query Time (converted from QPS)
    plot_load_balance_and_query_time(df_qtime, directory, m)
