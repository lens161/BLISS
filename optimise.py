import optuna # type: ignore
from multiprocessing import Process

from bliss import run_bliss
from config import Config

DATASET = "sift-128-euclidean"
EXP_NAME = "optimise"

def optimise_load_balance(bucket_size, learning_rate, batch_size, m, trial):
    conf = Config(dataset_name=DATASET, b=bucket_size, lr=learning_rate, m=m,
                  batch_size=batch_size)
    
    try:
        _, _, _, _, norm_entropy, _, _ = run_bliss(conf, mode="build", experiment_name=EXP_NAME, trial=trial)
    except optuna.exceptions.TrialPruned as e:
        raise e

    return norm_entropy

def optimise_m(bucket_size, learning_rate, batch_size, m, trial):
    conf = Config(dataset_name=DATASET, b=bucket_size, lr=learning_rate,
                  batch_size=batch_size, m=m)
    
    try:
        recall, results, _ = run_bliss(conf, mode="query", experiment_name=EXP_NAME, trial=trial)
    except optuna.exceptions.TrialPruned as e:
        raise e
    
    
    dist_comps_total = sum(result[2] for result in results)
    dist_comps_avg = dist_comps_total / len(results)
    return recall, dist_comps_avg


def objective(trial):
    bucket_size = 4096
    learning_rate = 0.001
    batch_size = 1000
    m = trial.suggest_int('m', 5, 25)

    normalised_entropy = optimise_m(bucket_size, learning_rate, batch_size, m, trial)
    return normalised_entropy

def create_study_load_balance(name):
    study = optuna.create_study(
        storage="sqlite:///opt_bliss.db",
        study_name=name,
        direction='maximize',
        load_if_exists=True
    )
    study.optimize(objective, n_trials=20)

def create_study_m(name):
    study = optuna.create_study(
        storage="sqlite:///opt_bliss.db",
        study_name=name,
        directions = ['maximize', 'minimize'],
        load_if_exists=True
    )
    study.optimize(objective, n_trials=20)

if __name__ == '__main__':
    optuna.delete_study(study_name="find_m_for_sift", storage="sqlite:///opt_bliss.db")
    STUDY_NAME = "find_m_for_sift"

    create_study_m(STUDY_NAME)