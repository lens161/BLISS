import optuna # type: ignore
from multiprocessing import Process

from bliss import run_bliss
from config import Config

DATASET = "sift-128-euclidean"
EXP_NAME = "optimise"

def optimise_bliss(bucket_size, learning_rate, batch_size,  m, trial):

    conf = Config(dataset_name=DATASET, b=bucket_size, lr=learning_rate, batch_size=batch_size, m=m)

    run_bliss(conf, mode="build", experiment_name=EXP_NAME)

    recall, results, _ = run_bliss(conf, mode="query", experiment_name=EXP_NAME, trial=trial)

    dist_comps_total = 0
    for result in results:
        dist_comps_total+=result[1]
    dist_comps_avg = dist_comps_total/len(results)
    return recall, dist_comps_avg

def objective(trial):
    bucket_size = trial.suggest_categorical('B', [1024, 2048, 4096, 8192])
    learning_rate = trial.suggest_float('lr', 0.001, 0.02)
    batch_size = trial.suggest_categorical('batch_size', [1000, 2000, 3000, 4000, 5000])
    m = trial.suggest_int('m', 5, 20)

    recall, dist_comps = optimise_bliss(bucket_size, learning_rate, batch_size, m, trial)
    return recall, dist_comps

def create_study(name):
    study = optuna.create_study(
        storage="sqlite:///opt_bliss.db",
        study_name = name,
        directions=['maximize', 'minimize'],
        load_if_exists=True
    )
    return study

def run_optimisation(name):
    study = optuna.load_study(
        storage="sqlite:///opt_bliss.db",
        study_name = name
    )
    study.optimize(objective, n_trials=10)

if __name__ == '__main__':
    STUDY_NAME = "find_hyperparams_v1"

    create_study(STUDY_NAME)
    
    processes = []
    
    for _ in range(4):
        p = Process(target=run_optimisation, args=(STUDY_NAME,))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
