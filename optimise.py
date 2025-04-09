import optuna # type: ignore
from multiprocessing import Process

from bliss import run_bliss
from config import Config

DATASET = "sift-128-euclidean"
EXP_NAME = "optimise"

def optimise_load_balance(bucket_size, learning_rate, batch_size, trial):
    conf = Config(dataset_name=DATASET, b=bucket_size, lr=learning_rate,
                  batch_size=batch_size)
    
    try:
        _, _, _, _, _, load_balance, _, _ = run_bliss(conf, mode="build", experiment_name=EXP_NAME, trial=trial)
    except optuna.exceptions.TrialPruned as e:
        raise e

    return load_balance

def optimise_m(bucket_size, learning_rate, batch_size, m, trial):
    conf = Config(dataset_name=DATASET, b=bucket_size, lr=learning_rate,
                  batch_size=batch_size, m=m)
    
    try:
        recall, results, _ = run_bliss(conf, mode="query", experiment_name=EXP_NAME, trial=trial)
    except optuna.exceptions.TrialPruned as e:
        raise e

    dist_comps_total = sum(result[1] for result in results)
    dist_comps_avg = dist_comps_total / len(results)
    return recall, dist_comps_avg


def objective(trial):
    bucket_size = trial.suggest_categorical('B', [2048, 4096])
    learning_rate = trial.suggest_categorical('lr', [0.001, 0.002, 0.005, 0.01])
    batch_size = trial.suggest_categorical('batch_size', [1000, 2000, 3000, 4000, 5000])

    load_balance = optimise_load_balance(bucket_size, learning_rate, batch_size, trial)
    return load_balance

def create_study_load_balance(name):
    study = optuna.create_study(
        storage="sqlite:///opt_bliss.db",
        study_name=name,
        direction='maximize',
        load_if_exists=True
    )
    study.optimize(objective, n_trials=25)

# def run_optimisation(name):
#     study = optuna.load_study(
#         storage="sqlite:///opt_bliss.db",
#         study_name = name
#     )
#     study.optimize(objective, n_trials=10)

# def run_optimisation(name):
#     study = optuna.load_study(
#         storage="sqlite:///opt_bliss.db",
#         study_name = name
#     )
#     study.optimize(objective, n_trials=10)

if __name__ == '__main__':
    STUDY_NAME = "find_b_for_sift"

    create_study_load_balance(STUDY_NAME)
    
    # processes = []
    
    # for _ in range(4):
    #     p = Process(target=run_optimisation, args=(STUDY_NAME,))
    #     p.start()
    #     processes.append(p)

    # for p in processes:
    #     p.join()
