import ast
import glob
import os
import matplotlib
import pandas as pd
import matplotlib.pyplot as plt

def process_load_balance(load_balance_str):
    try:
        load_balance_list = ast.literal_eval(load_balance_str)
        last_values = [sublist[-1] for sublist in load_balance_list if isinstance(sublist, list)]
        return sum(last_values) / len(last_values) if last_values else float('nan')
    except Exception as e:
        print(f"Error parsing load balance: {e}")
        return float('nan')

def collect_data_for_plot(hdf5_dir, m, include_qps=False):
    data = []

    # Load build.csv
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
                    batch_size = int(row['bs'])
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
                        (build_df['batch_size'] == batch_size)
                    ]

                    if match.empty or len(match) > 1:
                        print(f"Warning: No unique match for e={e}, i={i}, batch_size={batch_size}")
                        continue

                    load_balance_str = match.iloc[0]['load_balances']
                    load_balance = process_load_balance(load_balance_str)

                    data.append((batch_size, recall, qps, query_time_ms, load_balance))

        except Exception as e:
            print(f"Error processing {h5_file}: {e}")

    return pd.DataFrame(data, columns=['batch_size', 'recall', 'qps', 'query_time_ms', 'load_balance'])

def plot_batch_size_dual_y(df, directory, m, metric='recall'):
    if df.empty:
        print("No data to plot.")
        return

    df = df.sort_values(by='batch_size')

    fig, ax1 = plt.subplots(figsize=(8, 6))

    # --- Left Y-axis: Load Balance ---
    color1 = 'tab:blue'
    ax1.set_xlabel('Batch size', fontsize=22, labelpad=15)
    ax1.set_ylabel('Load balance', color=color1, fontsize=22, labelpad=15)

    # Convert Load Balance to scientific notation
    ax1.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

    line1, = ax1.plot(df['batch_size'], df['load_balance'], color=color1, marker='x', linestyle='--',
                      label='Load balance', linewidth=3, markersize=10)
    ax1.tick_params(axis='y', labelcolor=color1, labelsize=16)
    ax1.tick_params(axis='x', labelsize=16)

    # Set left y-axis limits: start at 0, upper limit dynamic
    load_balance_max = df['load_balance'].max()
    if pd.notna(load_balance_max):
        ax1.set_ylim(0, load_balance_max * 1.1)  # 10% padding on top

    # --- Right Y-axis: Recall, QPS, or Query Time ---
    ax2 = ax1.twinx()

    if metric == 'qps':
        color2 = 'tab:olive'
        metric_label = 'QPS'
        metric_data = df['qps']
        legend_label = 'QPS'
        title_label = 'QPS'
    elif metric == 'query_time':
        color2 = 'tab:green'
        metric_label = 'Query time (ms/query)'
        metric_data = df['query_time_ms']
        legend_label = 'Query time'
        title_label = 'query time'
    else:  # Default to 'recall'
        color2 = 'tab:red' 
        metric_label = 'Recall'
        metric_data = df['recall']
        legend_label = 'Recall'
        title_label = 'recall'

    ax2.set_ylabel(metric_label, color=color2, fontsize=22, labelpad=15)
    line2, = ax2.plot(df['batch_size'], metric_data, color=color2, marker='o', label=legend_label,
                      linewidth=3, markersize=10)
    ax2.tick_params(axis='y', labelcolor=color2, labelsize=16)

    # Set right y-axis limits: start at 0, upper limit dynamic
    metric_max = metric_data.max()
    if pd.notna(metric_max):
        ax2.set_ylim(0, metric_max * 1.1)  # 10% padding on top

    # --- Legend ---
    lines = [line1, line2]
    labels = [line.get_label() for line in lines]
    ax2.legend(lines, labels, loc='lower right', fontsize=14)

    # --- Title & Save ---
    fig.suptitle(f'Balance and {title_label} vs batch size (m={m})', fontsize=24)
    fig.tight_layout()

    ax1.grid(False)

    filename = f"batch_size_vs_load_balance_and_{metric.lower()}_m_{m}.svg"
    output_path = os.path.join(directory, filename)
    plt.savefig(output_path, format='svg', bbox_inches='tight')
    plt.close()
    print(f"Plot saved to: {output_path}")

# --- Main Usage ---
if __name__ == "__main__":
    directory = "results/glove_diff_batchsizes_v7"  # Adjust as needed
    m = 15

    # Plot Recall version
    df_recall = collect_data_for_plot(directory, m, include_qps=False)
    plot_batch_size_dual_y(df_recall, directory, m, metric='recall')

    # Plot QPS version
    df_qps = collect_data_for_plot(directory, m, include_qps=True)
    plot_batch_size_dual_y(df_qps, directory, m, metric='qps')

    # Plot Query Time version (derived from QPS)
    df_query_time = collect_data_for_plot(directory, m, include_qps=True)
    plot_batch_size_dual_y(df_query_time, directory, m, metric='query_time')
