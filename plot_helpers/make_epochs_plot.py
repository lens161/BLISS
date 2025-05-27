import ast
import glob
import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

# Process load balance string
def process_load_balance(load_balance_str):
    try:
        load_balance_list = ast.literal_eval(load_balance_str)
        last_values = [sublist[-1] for sublist in load_balance_list if isinstance(sublist, list)]
        return sum(last_values) / len(last_values) if last_values else float('nan')
    except Exception as e:
        print(f"Error parsing load balance: {e}")
        return float('nan')

# Collect data for the plot
def collect_data_for_plot(hdf5_dir, m, include_qps=False):
    data = []

    build_files = glob.glob(os.path.join(hdf5_dir, "*_build.csv"))
    if not build_files:
        raise FileNotFoundError("No build.csv file found.")
    build_df = pd.read_csv(build_files[0])

    hdf5_files = glob.glob(os.path.join(hdf5_dir, "*.h5"))

    for h5_file in hdf5_files:
        try:
            with pd.HDFStore(h5_file, mode='r') as store:
                averages_df = store['averages']
                m_data = averages_df[averages_df['m'] == m]

                if m_data.empty:
                    continue

                for _, row in m_data.iterrows():
                    e = int(row['e'])
                    i = int(row['i'])

                    if include_qps:
                        qps = float(row['qps'])
                        query_time_ms = 1000 / qps if qps > 0 else float('nan')
                        recall = None
                    else:
                        recall = float(row['avg_recall'])
                        query_time_ms = None
                        qps = None

                    match = build_df[
                        (build_df['epochs_per_it'] == e) & 
                        (build_df['iterations'] == i)
                    ]

                    if match.empty or len(match) > 1:
                        print(f"Warning: No unique match for e={e}, i={i}")
                        continue

                    load_balance_str = match.iloc[0]['load_balances']
                    load_balance = process_load_balance(load_balance_str)

                    data.append((e * i, recall, qps, query_time_ms, load_balance))

        except Exception as e:
            print(f"Error processing {h5_file}: {e}")

    return pd.DataFrame(data, columns=['epoch_count', 'recall', 'qps', 'query_time_ms', 'load_balance'])

# Unified plot function
def plot_load_balance_and_metric(df, directory, m, metric='recall'):
    """Plot load balance and either recall, query time, or qps vs epoch count."""
    df = df.sort_values(by='epoch_count')

    fig, ax1 = plt.subplots(figsize=(8, 6))

    ax1.tick_params(axis='x', labelsize=14)
    ax1.xaxis.set_major_locator(MultipleLocator(5))

    color1 = 'tab:blue'
    ax1.set_xlabel('Epochs', fontsize=22, labelpad=15)  # Increased font size
    ax1.set_ylabel('Load balance', color=color1, fontsize=22, labelpad=15)  # Increased font size
    line1, = ax1.plot(df['epoch_count'], df['load_balance'], color=color1, marker='x', linestyle='--',
                      label='Load balance', linewidth=3, markersize=10, markeredgewidth=2)  # Increased line thickness and marker size
    ax1.tick_params(axis='y', labelcolor=color1, labelsize=16)  # Increased font size for Y-axis
    ax1.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

    ax2 = ax1.twinx()

    if metric == 'qps':
        color2 = 'tab:red'
        ax2.set_ylabel('QPS', color=color2, fontsize=22, labelpad=15)  # Increased font size
        line2, = ax2.plot(df['epoch_count'], df['qps'], color=color2, marker='o', label='QPS',  # Legend as QPS
                          linewidth=3, markersize=10, markeredgewidth=2)  # Increased line thickness and marker size
    elif metric == 'query_time':
        color2 = 'tab:red'
        ax2.set_ylabel('Query time (ms/query)', color=color2, fontsize=22, labelpad=15)  # Full label for Y-axis
        line2, = ax2.plot(df['epoch_count'], df['query_time_ms'], color=color2, marker='o', label='Query time',  # Legend as Query time
                          linewidth=3, markersize=10, markeredgewidth=2)  # Increased line thickness and marker size
    elif metric == 'recall':
        color2 = 'tab:red'
        ax2.set_ylabel(f'Recall (m={m})', color=color2, fontsize=22, labelpad=15)  # Increased font size
        line2, = ax2.plot(df['epoch_count'], df['recall'], color=color2, marker='o', label=f'Recall (m={m})',  # Legend as Recall
                          linewidth=3, markersize=10, markeredgewidth=2)  # Increased line thickness and marker size

    ax2.tick_params(axis='y', labelcolor=color2, labelsize=16)  # Increased font size for Y-axis

    lines = [line1, line2]
    labels = [line.get_label() for line in lines]

    ax2.legend(
        lines,
        labels,
        loc='center right',
        bbox_to_anchor=(1.00, 0.5),
        fontsize=14,  # Increased font size for legend
        borderpad=0.3,
        borderaxespad=1,
        framealpha=0.8,
        edgecolor='black',
        handlelength=2,
        handleheight=2,
    )

    # Set the title dynamically based on the metric
    if metric == 'qps':
        title_metric = 'QPS'
    elif metric == 'query_time':
        title_metric = 'query time'
    else:  # default is 'recall'
        title_metric = 'recall'

    fig.suptitle(f'Load balance and {title_metric} vs epochs', fontsize=24)  # Increased font size for title
    fig.tight_layout()

    filename = f"load_balance_and_{metric}_vs_epochs_m_{m}.svg"
    output_path = os.path.join(directory, filename)
    plt.savefig(output_path, format='svg', bbox_inches='tight')
    plt.close()
    print(f"Plot saved to: {output_path}")

# Main usage
if __name__ == "__main__":
    directory = "results/glove_diff_epochs_v6"  # Adjust as needed
    m = 15

    # Collect data for recall
    df_recall = collect_data_for_plot(directory, m, include_qps=False)
    # Plot recall
    plot_load_balance_and_metric(df_recall, directory, m, metric='recall')

    # Collect data for QPS
    df_qps = collect_data_for_plot(directory, m, include_qps=True)
    # Plot QPS
    plot_load_balance_and_metric(df_qps, directory, m, metric='qps')

    # Collect data for Query time (derived from QPS)
    df_query_time = collect_data_for_plot(directory, m, include_qps=True)
    # Plot Query Time
    plot_load_balance_and_metric(df_query_time, directory, m, metric='query_time')
