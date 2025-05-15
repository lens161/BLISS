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
                    recall = None  # Not used if we're plotting QPS
                else:
                    qps = None  # Not used if we're plotting recall
                    recall = float(row['avg_recall'])

                # Match the parameters from averages with build.csv (match on epochs and iterations)
                match = build_df[
                    (build_df['epochs_per_it'] == e) & 
                    (build_df['iterations'] == i)
                ]
                
                # If there are no matches, or multiple matches, handle accordingly
                if match.empty:
                    print(f"Warning: No matching row in build.csv for epochs={e}, iterations={i} in {h5_file}")
                    continue
                elif len(match) > 1:
                    print(f"Warning: Multiple matches found in build.csv for epochs={e}, iterations={i} in {h5_file}")
                    continue

                # Extract load_balance from the matched row
                load_balance_str = match.iloc[0]['load_balances']
                load_balance = process_load_balance(load_balance_str)

            # Append the results
            data.append((epoch_count, recall, qps, load_balance))

        except Exception as e:
            print(f"Error processing {h5_file}: {e}")

    return pd.DataFrame(data, columns=['epoch_count', 'recall', 'qps', 'load_balance'])

def plot_load_balance_and_recall(df, directory, m):
    """Plot load balance and recall against epoch count with dual Y axes."""
    df = df.sort_values(by='epoch_count')

    fig, ax1 = plt.subplots(figsize=(7, 6))

    # x-axis
    ax1.tick_params(axis='x', labelsize=14)
    ax1.xaxis.set_major_locator(MultipleLocator(5))  # Set x-axis ticks every 5 units

    # --- Primary Y-axis: Load Balance ---
    color1 = 'tab:blue'
    ax1.set_xlabel('Epochs', fontsize=18, labelpad=15)
    ax1.set_ylabel('Load Balance', color=color1, fontsize=18, labelpad=15)
    line1, = ax1.plot(df['epoch_count'], df['load_balance'], color=color1, marker='x', linestyle='--', label='Load Balance', linewidth=3, markersize=8, markeredgewidth=2)
    ax1.tick_params(axis='y', labelcolor=color1, labelsize=14)
    ax1.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

    # --- Secondary Y-axis: Recall ---
    ax2 = ax1.twinx()
    color2 = 'tab:red'
    ax2.set_ylabel(f'Recall (m={m})', color=color2, fontsize=18, labelpad=15)
    line2, = ax2.plot(df['epoch_count'], df['recall'], color=color2, marker='o', label=f'Recall (m={m})', linewidth=3, markersize=8, markeredgewidth=2)
    ax2.tick_params(axis='y', labelcolor=color2, labelsize=14)

    # --- Legend ---
    lines = [line1, line2]
    labels = [line.get_label() for line in lines]

    ax2.legend(
        lines,
        labels,
        loc='center right',
        bbox_to_anchor=(1.00, 0.5),
        fontsize=12,  # Increase font size of the legend text
        title_fontsize=12,  # Font size for the title in the legend (if any)
        borderpad=0.3,  # Padding around the text inside the legend box
        borderaxespad=1,  # Distance between the legend and the axes
        framealpha=0.8,  # Transparency of the legend box
        edgecolor='black',  # Border color of the legend box
        handlelength=2,  # Length of the legend handles (the lines in the legend)
        handleheight=2,  # Height of the legend handles (the lines in the legend)
    )

    fig.suptitle(f'Load Balance and Recall (m={m}) vs Epochs', fontsize = 20)
    fig.tight_layout()
    plt.grid(True)

    # Save the plot as SVG instead of PNG
    output_path = os.path.join(directory, "load_balance_and_recall_per_epochs.svg")
    plt.savefig(output_path, format='svg', bbox_inches='tight')
    plt.close()
    print(f"Plot saved to: {output_path}")

def plot_load_balance_and_qps(df, directory, m):
    """Plot load balance and QPS against epoch count with dual Y axes."""
    df = df.sort_values(by='epoch_count')

    fig, ax1 = plt.subplots(figsize=(7, 6))

    # x-axis
    ax1.tick_params(axis='x', labelsize=14)
    ax1.xaxis.set_major_locator(MultipleLocator(5))  # Set x-axis ticks every 5 units

    # --- Primary Y-axis: Load Balance ---
    color1 = 'tab:blue'
    ax1.set_xlabel('Epochs', fontsize=18, labelpad=15)
    ax1.set_ylabel('Load Balance', color=color1, fontsize=18, labelpad=15)
    line1, = ax1.plot(df['epoch_count'], df['load_balance'], color=color1, marker='x', linestyle='--', label='Load Balance', linewidth=3, markersize=8, markeredgewidth=2)
    ax1.tick_params(axis='y', labelcolor=color1, labelsize=14)
    ax1.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

    # --- Secondary Y-axis: QPS ---
    ax2 = ax1.twinx()
    color2 = 'tab:red'
    ax2.set_ylabel(f'QPS', color=color2, fontsize=18, labelpad=15)
    line2, = ax2.plot(df['epoch_count'], df['qps'], color=color2, marker='o', label='QPS', linewidth=3, markersize=8, markeredgewidth=2)
    ax2.tick_params(axis='y', labelcolor=color2, labelsize=14)

    # --- Legend ---
    lines = [line1, line2]
    labels = [line.get_label() for line in lines]

    ax2.legend(
        lines,
        labels,
        loc='center right',
        bbox_to_anchor=(1.00, 0.5),
        fontsize=12,  # Increase font size of the legend text
        title_fontsize=12,  # Font size for the title in the legend (if any)
        borderpad=0.3,  # Padding around the text inside the legend box
        borderaxespad=1,  # Distance between the legend and the axes
        framealpha=0.8,  # Transparency of the legend box
        edgecolor='black',  # Border color of the legend box
        handlelength=2,  # Length of the legend handles (the lines in the legend)
        handleheight=2,  # Height of the legend handles (the lines in the legend)
    )

    fig.suptitle(f'Load Balance and QPS vs Epochs', fontsize = 20)
    fig.tight_layout()
    plt.grid(True)

    # Save the plot as SVG instead of PNG
    output_path = os.path.join(directory, "load_balance_and_qps_per_epochs.svg")
    plt.savefig(output_path, format='svg', bbox_inches='tight')
    plt.close()
    print(f"Plot saved to: {output_path}")


# Main usage
if __name__ == "__main__":
    directory = "results/glove_diff_epochs_v5_nomem"  # Adjust as needed
    m = 15

    # Collect data for both plots
    df_recall = collect_data_for_plot(directory, m, include_qps=False)
    df_qps = collect_data_for_plot(directory, m, include_qps=True)

    # Plot for Recall
    plot_load_balance_and_recall(df_recall, directory, m)

    # Plot for QPS
    plot_load_balance_and_qps(df_qps, directory, m)
