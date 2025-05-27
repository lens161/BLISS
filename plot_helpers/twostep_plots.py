import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def collect_metric_data_with_baseline(base_dir):
    dataset_metrics = {}  # {compression: {twostep_limit: (recall, qps)}}
    baseline = None

    subdirs = sorted([d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))])
    compression_dirs = [d for d in subdirs if d != "base"]

    # Get compression data
    for comp in compression_dirs:
        comp_path = os.path.join(base_dir, comp)
        h5_files = glob.glob(os.path.join(comp_path, "*.h5"))
        metric_data = {}

        for h5_file in h5_files:
            try:
                with pd.HDFStore(h5_file, mode='r') as store:
                    df = store['averages']
                    grouped = df.groupby('twostep_limit')

                    for limit, group in grouped:
                        recall = group['avg_recall'].mean()
                        qps = group['qps'].mean()

                        if limit not in metric_data:
                            metric_data[limit] = {'recall': [], 'qps': []}
                        metric_data[limit]['recall'].append(recall)
                        metric_data[limit]['qps'].append(qps)
            except Exception as e:
                print(f"Error processing {h5_file}: {e}")

        final_metrics = {limit: (np.mean(vals['recall']), np.mean(vals['qps']))
                         for limit, vals in metric_data.items()}

        dataset_metrics[comp] = final_metrics

    # Get baseline
    base_path = os.path.join(base_dir, "base")
    base_files = glob.glob(os.path.join(base_path, "*.h5"))
    recall_vals = []
    qps_vals = []

    for h5_file in base_files:
        try:
            with pd.HDFStore(h5_file, mode='r') as store:
                df = store['averages']
                recall_vals.extend(df['avg_recall'].tolist())
                qps_vals.extend(df['qps'].tolist())
        except Exception as e:
            print(f"Error reading baseline {h5_file}: {e}")

    if recall_vals and qps_vals:
        baseline = (np.mean(recall_vals), np.mean(qps_vals))

    return dataset_metrics, baseline


def plot_split_barplots(metrics_by_compression, baseline, dataset_name, output_dir, exp_type):
    twostep_limits = sorted({t for m in metrics_by_compression.values() for t in m})
    compressions = sorted(metrics_by_compression.keys(), key=lambda x: int(''.join(filter(str.isdigit, x))))

    num_subplots = len(compressions)
    fig, axes = plt.subplots(1, num_subplots, figsize=(5 * num_subplots, 6), sharey=False)

    if num_subplots == 1:
        axes = [axes]  # Ensure iterable

    recall_color = 'C3'  # red
    qps_color = 'C9'     # cyan

    for ax, comp in zip(axes, compressions):
        metrics = metrics_by_compression[comp]
        recall_vals = [metrics.get(limit, (0, 0))[0] for limit in twostep_limits]
        qps_vals = [metrics.get(limit, (0, 0))[1] for limit in twostep_limits]

        x = np.arange(len(twostep_limits)) * 0.6  # Reduce spacing between bar groups
        bar_width = 0.25

        ax1 = ax
        ax2 = ax1.twinx()

        ax1.bar(x - bar_width / 2, recall_vals, width=bar_width, color=recall_color, label='Recall')
        ax2.bar(x + bar_width / 2, qps_vals, width=bar_width, color=qps_color, label='QPS')

        # Add baseline lines
        if baseline:
            base_recall, base_qps = baseline
            baseline_recall_color = recall_color  # use same red for baseline recall line
            baseline_qps_color = qps_color        # use same yellow for baseline QPS line

            ax1.axhline(base_recall, color=baseline_recall_color, linestyle='--', linewidth=2, label='Baseline Recall')
            ax2.axhline(base_qps, color=baseline_qps_color, linestyle='--', linewidth=2, label='Baseline QPS')

        ax1.set_xticks(x)
        ax1.set_xticklabels([str(t) for t in twostep_limits], rotation=45, ha='right')
        if ax == axes[1]:
            ax1.set_xlabel("Threshold", fontsize=18)

        if ax == axes[0]:
            ax1.set_ylabel("Recall", color=recall_color, fontsize=18)
        if ax == axes[-1]:
            ax2.set_ylabel("QPS", color=qps_color, fontsize=18)
        ax1.tick_params(axis='x', labelsize=14)
        # Only show recall y-axis ticks and labels on the leftmost subplot
        if ax == axes[0]:
            ax1.tick_params(axis='y', labelcolor=recall_color, labelsize=14)
        else:
            ax1.tick_params(axis='y', labelleft=False)

        # Only show QPS y-axis ticks and labels on the rightmost subplot
        if ax == axes[-1]:
            ax2.tick_params(axis='y', labelcolor=qps_color, labelsize=14)
        else:
            ax2.tick_params(axis='y', labelright=False)
        ax.set_title(f"{comp}", fontsize=14)
        if dataset_name == "Deep1B":
            ax1.set_ylim((0.0, 1.0))
            ax2.set_ylim((0.0, 3.85))
        elif dataset_name == "Yandex":
            ax1.set_ylim((0.0, 0.4))
            ax2.set_ylim((0.0, 2.4))


    handles = [
        plt.Line2D([0], [0], color=recall_color, lw=4, label='Recall'),
        plt.Line2D([0], [0], color=qps_color, lw=4, label='QPS'),
        plt.Line2D([0], [0], color=baseline_recall_color, lw=2, linestyle='--', label='Baseline Recall'),
        plt.Line2D([0], [0], color=baseline_qps_color, lw=2, linestyle='--', label='Baseline QPS'),
    ]
    fig.suptitle(f"Recall and QPS by threshold for {dataset_name}", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.95])  # Leave more space at the top (reduce top to 0.85)
    fig.subplots_adjust(wspace=0.05)  # Smaller wspace means less horizontal whitespace
    fig.legend(
        handles=handles,
        loc='upper right',
        bbox_to_anchor=(0.95, 0.99),  # Move legend left and align to top of figure
        ncol=2,
        fontsize=12
    )

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{exp_type}_{dataset_name}_split_barplot_with_baseline.svg")
    plt.savefig(output_path, format='svg', bbox_inches='tight')
    plt.close()
    print(f"Saved plot to {output_path}")


if __name__ == "__main__":
    exp_type = "twostep_fixed"
    root_dir = f"results/{exp_type}"  # Contains dataset subfolders
    output_dir = "plots"

    dataset_dirs = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d)) and d != "base"]

    for dataset in dataset_dirs:
        full_path = os.path.join(root_dir, dataset)
        data, baseline = collect_metric_data_with_baseline(full_path)
        plot_split_barplots(data, baseline, dataset_name=dataset, output_dir=root_dir, exp_type=exp_type)