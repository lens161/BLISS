import ast
import glob
import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

# -----------------------------
# Utilities
# -----------------------------

def process_load_balance(load_balance_str):
    try:
        load_balance_list = ast.literal_eval(load_balance_str)
        last_values = [sublist[-1] for sublist in load_balance_list if isinstance(sublist, list)]
        return sum(last_values) / len(last_values) if last_values else float('nan')
    except Exception as e:
        print(f"Error parsing load balance: {e}")
        return float('nan')

def read_loss_values(directory, experiment_prefix):
    loss_file_pattern = os.path.join(directory, f"{experiment_prefix}*.txt")
    loss_files = glob.glob(loss_file_pattern)

    if not loss_files:
        raise FileNotFoundError(f"No loss file found with prefix {experiment_prefix}")
    if len(loss_files) > 1:
        print(f"Warning: Multiple loss files found. Using the first: {loss_files[0]}")

    with open(loss_files[0], 'r') as f:
        losses = [float(line.strip()) for line in f if line.strip()]
    
    return losses

def create_loss_df(losses):
    return pd.DataFrame({
        'epoch': list(range(1, len(losses) + 1)),
        'loss': losses
    })

def merge_loss_with_data(df, losses):
    if len(df) != len(losses):
        raise ValueError("Mismatch between number of rows in DataFrame and number of loss values.")
    
    df = df.copy()
    df['loss'] = losses
    return df

# -----------------------------
# Data Collection
# -----------------------------

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

# -----------------------------
# Plotting
# -----------------------------

def plot_loss_and_recall(recall_df, loss_df, directory, m):
    recall_df = recall_df.sort_values(by='epoch_count')
    loss_df = loss_df.sort_values(by='epoch')

    fig, ax1 = plt.subplots(figsize=(8, 6))
    ax1.tick_params(axis='x', labelsize=14)
    ax1.xaxis.set_major_locator(MultipleLocator(5))

    # Plot full loss curve (purple)
    color1 = 'tab:purple'
    ax1.set_xlabel('Epochs', fontsize=22, labelpad=15)
    ax1.set_ylabel('Loss', color=color1, fontsize=22, labelpad=15)
    line1 = ax1.plot(loss_df['epoch'], loss_df['loss'], color=color1, linestyle='-', marker='x',
                     label='Loss', linewidth=2, markersize=6, markeredgewidth=1.5)[0]
    ax1.tick_params(axis='y', labelcolor=color1, labelsize=16)

    # Set y-limits for loss axis starting from 0
    loss_max = loss_df['loss'].max()
    if pd.notna(loss_max):
        ax1.set_ylim(0, loss_max * 1.1)

    ax2 = ax1.twinx()

    # Plot recall with connecting lines and markers (red)
    color2 = 'tab:red'
    ax2.set_ylabel(f'Recall (m={m})', color=color2, fontsize=22, labelpad=15)
    line2 = ax2.plot(recall_df['epoch_count'], recall_df['recall'], color=color2, linestyle='-',
                     marker='o', label=f'Recall (m={m})', linewidth=2, markersize=8, markeredgewidth=2)[0]
    ax2.tick_params(axis='y', labelcolor=color2, labelsize=16)

    # Set y-limits for recall axis starting from 0
    recall_max = recall_df['recall'].max()
    if pd.notna(recall_max):
        ax2.set_ylim(0, recall_max * 1.1)

    # Legend
    lines = [line1, line2]
    labels = [line.get_label() for line in lines]
    ax2.legend(lines, labels, loc='center right', bbox_to_anchor=(1.00, 0.5), fontsize=14,
               borderpad=0.3, borderaxespad=1, framealpha=0.8, edgecolor='black',
               handlelength=2, handleheight=2)

    fig.suptitle(f'Loss and Recall vs Epochs', fontsize=24)
    fig.tight_layout()
    filename = f"loss_and_recall_vs_epochs_m_{m}.svg"
    output_path = os.path.join(directory, filename)
    plt.savefig(output_path, format='svg', bbox_inches='tight')
    plt.close()
    print(f"Plot saved to: {output_path}")


def plot_load_balance_and_metric(df, directory, m, metric='recall'):
    df = df[df['epoch_count'] > 0].copy()  # Filter out epoch_count == 0
    df = df.sort_values(by='epoch_count')

    fig, ax1 = plt.subplots(figsize=(8, 6))

    ax1.tick_params(axis='x', labelsize=14)
    ax1.xaxis.set_major_locator(MultipleLocator(5))

    color1 = 'tab:blue'
    ax1.set_xlabel('Epochs', fontsize=22, labelpad=15)
    ax1.set_ylabel('Load balance', color=color1, fontsize=22, labelpad=15)
    line1 = ax1.plot(df['epoch_count'], df['load_balance'], color=color1, marker='x', linestyle='--',
                     label='Load balance', linewidth=3, markersize=10, markeredgewidth=2)[0]
    ax1.tick_params(axis='y', labelcolor=color1, labelsize=16)
    ax1.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

    # Set y-limits for load balance axis starting at 0
    load_balance_max = df['load_balance'].max()
    if pd.notna(load_balance_max):
        ax1.set_ylim(0, load_balance_max * 1.1)

    ax2 = ax1.twinx()

    if metric == 'qps':
        color2 = 'tab:olive'
        ax2.set_ylabel('QPS', color=color2, fontsize=22, labelpad=15)
        line2 = ax2.plot(df['epoch_count'], df['qps'], color=color2, marker='o', label='QPS',
                         linewidth=3, markersize=10, markeredgewidth=2)[0]
        metric_data = df['qps']
    elif metric == 'query_time':
        color2 = 'tab:green'
        ax2.set_ylabel('Query time (ms/query)', color=color2, fontsize=22, labelpad=15)
        line2 = ax2.plot(df['epoch_count'], df['query_time_ms'], color=color2, marker='o', label='Query time',
                         linewidth=3, markersize=10, markeredgewidth=2)[0]
        metric_data = df['query_time_ms']
    elif metric == 'recall':
        color2 = 'tab:red'
        ax2.set_ylabel(f'Recall (m={m})', color=color2, fontsize=22, labelpad=15)
        line2 = ax2.plot(df['epoch_count'], df['recall'], color=color2, marker='o', label=f'Recall (m={m})',
                         linewidth=3, markersize=10, markeredgewidth=2)[0]
        metric_data = df['recall']

    ax2.tick_params(axis='y', labelcolor=color2, labelsize=16)

    # Set y-limits for metric axis starting at 0
    metric_max = metric_data.max()
    if pd.notna(metric_max):
        ax2.set_ylim(0, metric_max * 1.1)

    lines = [line1, line2]
    labels = [line.get_label() for line in lines]

    ax2.legend(
        lines,
        labels,
        loc='center right',
        bbox_to_anchor=(1.00, 0.5),
        fontsize=14,
        borderpad=0.3,
        borderaxespad=1,
        framealpha=0.8,
        edgecolor='black',
        handlelength=2,
        handleheight=2,
    )

    title_metric = {'qps': 'QPS', 'query_time': 'query time'}.get(metric, 'recall')
    fig.suptitle(f'Load balance and {title_metric} vs epochs', fontsize=24)
    fig.tight_layout()

    filename = f"load_balance_and_{metric}_vs_epochs_m_{m}.svg"
    output_path = os.path.join(directory, filename)
    plt.savefig(output_path, format='svg', bbox_inches='tight')
    plt.close()
    print(f"Plot saved to: {output_path}")

# -----------------------------
# Main Execution
# -----------------------------

if __name__ == "__main__":
    directory = "results/glove_diff_epochs_v7"
    m = 15
    experiment_prefix = "glove_diff_epochs_v7"

    # Load recall data
    df_recall = collect_data_for_plot(directory, m, include_qps=False)

    # Load full loss curve
    loss_values = read_loss_values(directory, experiment_prefix)
    df_loss = create_loss_df(loss_values)

    # Plot all loss values + recall points
    plot_loss_and_recall(df_recall, df_loss, directory, m)

    # QPS + Load balance
    df_qps = collect_data_for_plot(directory, m, include_qps=True)
    plot_load_balance_and_metric(df_qps, directory, m, metric='qps')

    # Query time + Load balance
    df_query_time = collect_data_for_plot(directory, m, include_qps=True)
    plot_load_balance_and_metric(df_query_time, directory, m, metric='query_time')
