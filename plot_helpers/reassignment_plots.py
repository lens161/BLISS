import matplotlib.pyplot as plt
import os
import pandas as pd


def plot_time_vs_chunk_size(experiment_name, hardware = 'hp', time = 'build'):
    """
    Set hardware = 'hp' for hp nodes
    Set hardware = 'desk' for desktops
    Read a build-log CSV and plot chunk size vs total build time (in hours),
    with separate lines for each reass_mode found in the data.
    """

    time_metric = 'minutes' if time == 'train' else 'hours'

    if hardware == 'hp':
        csv_file = f"results/{experiment_name}/{experiment_name}_build.csv"
        out_dir = f"results/{experiment_name}"
    elif hardware == 'desk':
        csv_file = f"results/{experiment_name}_desk/{experiment_name}_build.csv"
        out_dir = f"results/{experiment_name}_desk"
    else:
        print("no hardware type given. enter 'hp' or 'desk' ")


    df = pd.read_csv(csv_file)

    # convert seconds to hours
    df['build_time'] = df['build_time'] / 3600
    df['train_time'] = df['train_time_per_r'] / 60
    df['reass_time'] = df['final_assign_time_per_r'] / 3600

    os.makedirs(out_dir, exist_ok=True)

    plt.figure(figsize=(8, 5))
    for mode in sorted(df['reass_mode'].unique()):
        subset = df[df['reass_mode'] == mode].sort_values('reass_chunk_size')
        key = f'{time}_time'
        plt.plot(
            subset['reass_chunk_size'],
            subset[key],
            marker='x',
            linestyle='-',
            label=f"Mark {mode}"
        )

    name = time.capitalize()
    if time == 'reass':
        name = 'Final assignment'
    plt.xlabel("Chunk size")
    plt.ylabel(f"Total time ({time_metric})")
    if hardware == 'desk' :
        plt.title(f"{name} time vs chunk size (desktop)")
    elif hardware == 'hp': 
        plt.title(f"{name} time vs chunk size (high performance node)")
    plt.grid(True)
    plt.legend(title="Reassign mode")
    plt.tight_layout()

    plt.savefig(f"{out_dir}/{time}_time_vs_chunk_size_by_mode.svg", dpi=300)
    plt.close()

def plot_mem_vs_chunk_size(experiment_name):
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
            subset['ram_final_ass'],
            marker='o',
            linestyle='-',
            label=f"Mark {mode}"
        )

    plt.xlabel("Chunk size")
    plt.ylabel("RAM used during final assignemnt (MB)")
    plt.title(f"Build time vs chunk size")
    plt.grid(True)
    plt.legend(title="Reassign mode")
    plt.tight_layout()

    plt.savefig(f"{out_dir}/ram_vs_chunk_size_by_mode.png", dpi=300)
    plt.close()

def plot_vram_vs_chunk_size(experiment_name):
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
        if mode == 2:
            ls = ':'
        else: 
            ls = '-'
        plt.plot(
            subset['reass_chunk_size'],
            subset['vram_final_assignement'],
            marker='o',
            linestyle=ls,
            label=f"Mark {mode}"
        )

    plt.xlabel("Chunk size")
    plt.ylabel("VRAM used during final assignemnt (MB)")
    plt.title(f"Build time vs chunk size")
    plt.grid(True)
    plt.legend(title="Reassign mode")
    plt.tight_layout()

    plt.savefig(f"{out_dir}/vram_vs_chunk_size_by_mode.png", dpi=300)
    plt.close()

def ram_usage_latex_table(experiment_name):
    """
    Read the build-log CSV for the given experiment and hardware,
    compute average RAM during training and final assignment per reass_mode,
    and return a LaTeX table as a string.
    """

    csv_file = f"results/{experiment_name}/{experiment_name}_build.csv"
    df = pd.read_csv(csv_file)

    grouped = df.groupby('reass_mode').agg(
        avg_ram_train = ('ram_training', 'mean'),
        avg_ram_final = ('ram_final_ass', 'mean')
    ).reset_index()

    table  = "\\begin{tabular}{c r r}\n"
    table += "  \\toprule\n"
    table += "  Reassign Mode & Training RAM (MB) & Final Assign RAM (MB)\\\\\n"
    table += "  \\midrule\n"
    for _, row in grouped.iterrows():
        mode = int(row['reass_mode'])
        t_mem = row['avg_ram_train']
        f_mem = row['avg_ram_final']
        table += f"  {mode:^13d} & {t_mem:>17.2f} & {f_mem:>20.2f}\\\\\n"
    table += "  \\bottomrule\n"
    table += "\\end{tabular}\n"

    print(table)
    return table

def make_plots(experiment_name):
    '''
    Include all plot functions that should be run here.
    '''
    plot_time_vs_chunk_size(experiment_name,'hp', "train")
    plot_time_vs_chunk_size(experiment_name,'hp', "reass")
    plot_time_vs_chunk_size(experiment_name,'hp', "build")

if __name__ == "__main__":
    make_plots("some_exp")