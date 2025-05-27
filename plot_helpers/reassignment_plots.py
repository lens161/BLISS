import matplotlib.pyplot as plt
import os
import pandas as pd
import glob

COLORMAP={
    0: "blue",
    1: "orange",
    2: "green",
    3: "red"
}

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

def recall_dist_comps_latex_table(experiment_name):
    """
    Read all the .h5 stores in results/{experiment_name},
    compute average recall and average distance computations
    per (m, chunk_size), and print a LaTeX table with grid lines
    and a divider after each block of m.
    """

    folder = os.path.join("results", experiment_name)
    files = glob.glob(os.path.join(folder, "*.h5"))
    if not files:
        raise FileNotFoundError(f"No .h5 files in {folder}")
    results, averages = compile_query_results(files)

    # collect rows
    rows = []
    for avg_df, indiv_df in zip(averages, results):
        rows.append({
            'm':               int(avg_df['m'].iloc[0]),
            'chunk_size':      int(avg_df['chunk_size'].iloc[0]),
            'avg_recall':      float(avg_df['avg_recall'].iloc[0]),
            'avg_dist_comps':  indiv_df['distance_computations'].mean()
        })

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("No rows collected; check your HDF5 stores.")
    grp = df.groupby(['m','chunk_size'], as_index=False) \
            .agg(avg_recall=('avg_recall','mean'),
                 avg_dist_comps=('avg_dist_comps','mean')) \
            .sort_values(['m','chunk_size'])

    # build the LaTeX table
    table  = "\\begin{table}[ht]\n"
    table += "  \\centering\n"
    table += "  \\begin{tabular}{@{}c|c|c|c@{}}\n"
    table += "    \\multicolumn{1}{c}{\\textbf{$m$}}"
    table += " & \\multicolumn{1}{c}{\\textbf{Chunk size}}"
    table += " & \\multicolumn{1}{c}{\\textbf{Avg. Recall}}"
    table += " & \\multicolumn{1}{c}{\\textbf{Avg. \\# Dist. Comps}} \\\\\n"
    table += "    \\hline\n"

    last_m = None
    for _, row in grp.iterrows():
        m_i   = int(row['m'])
        cs_i  = int(row['chunk_size'])
        rec_f = row['avg_recall']
        rd_i  = int(round(row['avg_dist_comps']))

        # whenever m changes, insert a divider
        if last_m is not None and m_i != last_m:
            table += "    \\hline\n"

        table += f"    {m_i} & {cs_i} & {rec_f:.3f} & {rd_i} \\\\\n"
        last_m = m_i

    table += "    \\hline\n"
    table += "  \\end{tabular}\n"
    table += f"  \\caption{{Recall and mean \\# distance computations per $(m,\\text{{chunk size}})$ for {experiment_name}.}}\n"
    table += f"  \\label{{tab:recall_dist_comps_{experiment_name}}}\n"
    table += "\\end{table}\n"

    print(table)
    return table

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
        csv_file = f"results/{experiment_name}_desk/{experiment_name}_desk_build.csv"
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
            color = COLORMAP.get(mode),
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
            color = COLORMAP.get(mode),
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
            color = COLORMAP.get(mode),
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

def plot_vram_train_vs_final(experiment_name):
    """
    Plot VRAM during training and VRAM during final assignment vs. chunk size
    for one specific reass_mode.
    """

    reass_mode = 2
    csv_file = f"results/{experiment_name}/{experiment_name}_build.csv"
    out_dir  = f"results/{experiment_name}"
    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(csv_file)
    df_mode = df[df['reass_mode'] == reass_mode].sort_values('reass_chunk_size')

    x = df_mode['reass_chunk_size']
    y_train = df_mode['vram_training'] / 1024**3
    y_final = df_mode['vram_final_assignement'] / 1024**3

    plt.figure(figsize=(8,5))
    plt.plot(x, y_train, marker='x', linestyle='-', color = "blue", label='VRAM training')
    plt.plot(x, y_final, marker='x', linestyle='--', color = "red", label='VRAM final assignment')
    plt.xlabel("Chunk size")
    plt.ylabel("VRAM (GB)")
    plt.title(f"VRAM vs chunk size")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    filename = f"{out_dir}/vram_train_vs_final_mode{reass_mode}.svg"
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"Saved plot to {filename}")

def plot_mem_vram_final_ass(experiment_name):
    """
    Plot VRAM and RAM for final assignment (dashed and solid) vs. chunk size
    for every reass_mode in one single plot, excluding training data.
    """
    csv_file = f"results/{experiment_name}/{experiment_name}_build.csv"
    out_dir  = f"results/{experiment_name}"
    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(csv_file)
    modes = sorted(df['reass_mode'].unique())
    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

    plt.figure(figsize=(12, 7))
    for idx, mode in enumerate(modes):
        df_mode = df[df['reass_mode'] == mode].sort_values('reass_chunk_size')
        x = df_mode['reass_chunk_size']
        # Final-assignment VRAM and RAM (convert bytes to GB)
        y_vram_final = df_mode['vram_final_assignement'] / 1024**3
        y_ram_final  = df_mode['ram_final_ass']   / 1024

        color = color_cycle[idx % len(color_cycle)]

        # Plot final-assignment VRAM (dashed) and RAM (solid)
        plt.plot(
            x, y_ram_final,
            marker='s', linestyle='-', color=color,
            label=f"mode {mode} RAM final"
        )
        plt.plot(
            x, y_vram_final,
            marker='o', linestyle='--', color=color,
            label=f"mode {mode} VRAM final"
        )

    plt.xlabel("Chunk size")
    plt.ylabel("Memory (GB)")
    plt.title(f"{experiment_name}: Final-assignment RAM & VRAM vs. chunk size (all modes)")
    plt.grid(True)
    plt.legend(loc='best', fontsize='small', ncol=2)
    plt.tight_layout()

    out_path = f"{out_dir}/mem_vram_final_all_modes.svg"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved final-assignment RAM & VRAM plot to {out_path}")

def plot_mem_vram_final_ass_sep(experiment_name):
    """
    Plot final-assignment RAM (per mode) and a single VRAM curve (common to all modes)
    vs. chunk size, saving them as two separate figures with enhanced formatting.
    """
    csv_file = f"results/{experiment_name}/{experiment_name}_build.csv"
    out_dir = f"results/{experiment_name}"
    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(csv_file)
    modes = sorted(df['reass_mode'].unique())
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    # --- RAM Plot ---
    fig_ram, ax_ram = plt.subplots(figsize=(8, 6))
    for idx, mode in enumerate(modes):
        df_mode = df[df['reass_mode'] == mode].sort_values('reass_chunk_size')
        x = df_mode['reass_chunk_size']
        y_ram = df_mode['ram_final_ass'] / 1024
        ls = ':' if mode == 2 or mode == 3 else '-'
        if mode == 0:
            c = 'orange'
        elif mode == 1:
            c = 'blue'
        else:
            c = colors[idx % len(colors)]
        ax_ram.plot(
            x, y_ram,
            color=c,
            marker='s', linestyle=ls,
            linewidth=3, markersize=10,
            label=f"Mark {mode}"
        )
    ax_ram.set_xlabel("Chunk size", fontsize=22, labelpad=15)
    ax_ram.set_ylabel("RAM (GB)", fontsize=22, labelpad=15)
    ax_ram.tick_params(axis='both', labelsize=16)
    ax_ram.grid(True)
    fig_ram.suptitle(f"RAM vs. chunk size", fontsize=24)
    ax_ram.legend(loc='best', fontsize=14)
    fig_ram.tight_layout()
    fig_ram.subplots_adjust(top=0.88)
    ram_path = f"{out_dir}/ram_final_all_modes.svg"
    fig_ram.savefig(ram_path, format='svg', bbox_inches='tight')
    plt.close(fig_ram)
    print(f"Saved RAM plot to {ram_path}")

    # --- VRAM Plot ---
    df_vram = (
        df
        .sort_values('reass_chunk_size')
        .drop_duplicates('reass_chunk_size', keep='first')
    )
    x_vram = df_vram['reass_chunk_size']
    y_vram = df_vram['vram_final_assignement'] / 1024**3

    fig_vram, ax_vram = plt.subplots(figsize=(8, 6))
    ax_vram.plot(
        x_vram, y_vram,
        color='tab:red',
        marker='o', linestyle='--',
        linewidth=3, markersize=10,
        label="VRAM (all Marks)"
    )
    ax_vram.set_xlabel("Chunk size", fontsize=22, labelpad=15)
    ax_vram.set_ylabel("VRAM (GB)", fontsize=22, labelpad=15)
    ax_vram.tick_params(axis='both', labelsize=16)
    ax_vram.grid(True)
    fig_vram.suptitle(f"VRAM vs. chunk size", fontsize=24)
    ax_vram.legend(loc='best', fontsize=14)
    fig_vram.tight_layout()
    fig_vram.subplots_adjust(top=0.88)
    vram_path = f"{out_dir}/vram_final_all_modes.svg"
    fig_vram.savefig(vram_path, format='svg', bbox_inches='tight')
    plt.close(fig_vram)
    print(f"Saved VRAM plot to {vram_path}")




def ram_usage_latex_table(experiment_name):
    """
    Read the build-log CSV for the given experiment and hardware,
    compute average RAM during training and final assignment per reass_mode,
    and return a fully centered LaTeX table in the ‘stacked’ style.
    """
    csv_file = f"results/{experiment_name}/{experiment_name}_build.csv"
    df = pd.read_csv(csv_file)
    grp = (df.groupby('reass_mode')
             .agg(avg_ram_train=('ram_training','mean'),
                  avg_ram_final=('ram_final_ass','mean'))
             .sort_index())

    modes  = list(grp.index)
    trains = [grp.loc[m, 'avg_ram_train'] for m in modes]
    finals = [grp.loc[m, 'avg_ram_final'] for m in modes]

    def stack_column(vals, fmt):
        lines = [fmt.format(v) for v in vals]
        inner = " \\\\ ".join(lines)
        return f"\\begin{{tabular}}[c]{{@{{}}c@{{}}}} {inner} \\end{{tabular}}"

    mode_col  = stack_column(modes,  "{:d}")
    train_col = stack_column(trains, "{:.2f}")
    final_col = stack_column(finals, "{:.2f}")

    table  = "\\begin{table}[ht]\n"
    table += "  \\centering\n"
    table += "  \\begin{tabular}{@{}c|c|c|c@{}}\n"
    table += "    \\multicolumn{1}{c}{\\textbf{Experiment}}"
    table += " & \\multicolumn{1}{c}{\\textbf{Mode}}"
    table += " & \\multicolumn{1}{c}{\\textbf{Train RAM (MB)}}"
    table += " & \\multicolumn{1}{c}{\\textbf{Final RAM (MB)}} \\\\\n"
    table += "    \\hline\n"
    table += f"    {experiment_name}"
    table += f" & {mode_col}"
    table += f" & {train_col}"
    table += f" & {final_col} \\\\\n"
    table += "  \\end{tabular}\n"
    table += "\\end{table}\n"

    print(table)
    return table

def make_plots(experiment_name):
    '''
    Include all plot functions that should be run here.
    '''
    # # make plots for whole experiment, add more plot functions as needed
    # plot_vram_vs_chunk_size(experiment_name)
    # plot_mem_vs_chunk_size(experiment_name)
    # ram_usage_latex_table(experiment_name)
    # plot_vram_train_vs_final(experiment_name)
    # recall_dist_comps_latex_table(experiment_name)
    plot_time_vs_chunk_size(experiment_name, "hp", "train")
    # plot_mem_vram_final_ass_sep(experiment_name)
    # plot_time_vs_chunk_size(experiment_name,'hp', "train")
    # plot_time_vs_chunk_size(experiment_name,'hp', "reass")
    # plot_time_vs_chunk_size(experiment_name,'hp', "build")

if __name__ == "__main__":
    make_plots("compare_time")