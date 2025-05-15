import ast
import glob
import os
import matplotlib
import pandas as pd
import matplotlib.pyplot as plt

matplotlib.rcParams.update({
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
    Collect batch_size, recall/qps, and load_balance for a specific m.
    """
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
                        recall = None
                    else:
                        recall = float(row['avg_recall'])
                        qps = None

                    match = build_df[
                        (build_df['batch_size'] == batch_size)
                    ]

                    if match.empty or len(match) > 1:
                        print(f"Warning: No unique match for e={e}, i={i}, batch_size={batch_size}")
                        continue

                    load_balance_str = match.iloc[0]['load_balances']
                    load_balance = process_load_balance(load_balance_str)

                    data.append((batch_size, recall, qps, load_balance))

        except Exception as e:
            print(f"Error processing {h5_file}: {e}")

    return pd.DataFrame(data, columns=['batch_size', 'recall', 'qps', 'load_balance'])

def plot_batch_size_dual_y(df, directory, m, include_qps=False):
    """Plot batch_size vs load_balance and recall (or QPS) with dual Y axes."""
    if df.empty:
        print("No data to plot.")
        return

    df = df.sort_values(by='batch_size')

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # --- Left Y-axis: Load Balance ---
    color1 = 'tab:blue'
    ax1.set_xlabel('Batch Size', fontsize=18, labelpad=15)
    ax1.set_ylabel('Load Balance', color=color1, fontsize=18, labelpad=15)
    line1, = ax1.plot(df['batch_size'], df['load_balance'], color=color1, marker='x', linestyle='--',
                      label='Load Balance', linewidth=2, markersize=8)
    ax1.tick_params(axis='y', labelcolor=color1, labelsize=14)
    ax1.tick_params(axis='x', labelsize=14)
    ax1.grid(True)

    # --- Right Y-axis: Recall or QPS ---
    ax2 = ax1.twinx()
    if include_qps:
        color2 = 'tab:green'
        metric_label = 'QPS'
        metric_data = df['qps']
    else:
        color2 = 'tab:red'
        metric_label = 'Recall'
        metric_data = df['recall']

    ax2.set_ylabel(metric_label, color=color2, fontsize=18, labelpad=15)
    line2, = ax2.plot(df['batch_size'], metric_data, color=color2, marker='o', label=metric_label,
                      linewidth=2, markersize=8)
    ax2.tick_params(axis='y', labelcolor=color2, labelsize=14)

    # --- Legend ---
    lines = [line1, line2]
    labels = [line.get_label() for line in lines]
    ax2.legend(lines, labels, loc='center right', fontsize=12)

    # --- Title & Save ---
    title_metric = 'QPS' if include_qps else 'Recall'
    fig.suptitle(f'Load Balance and {title_metric} vs Batch Size (m={m})', fontsize=20)
    fig.tight_layout()

    filename = f"batch_size_vs_load_balance_and_{title_metric.lower()}_m_{m}.svg"
    output_path = os.path.join(directory, filename)
    plt.savefig(output_path, format='svg', bbox_inches='tight')
    plt.close()
    print(f"Plot saved to: {output_path}")

# --- Main Usage ---
if __name__ == "__main__":
    directory = "results/glove_diff_batchsizes_v5_nomem"  # Adjust as needed
    m = 15

    # Plot Recall version
    df_recall = collect_data_for_plot(directory, m, include_qps=False)
    plot_batch_size_dual_y(df_recall, directory, m, include_qps=False)

    # Plot QPS version
    df_qps = collect_data_for_plot(directory, m, include_qps=True)
    plot_batch_size_dual_y(df_qps, directory, m, include_qps=True)
