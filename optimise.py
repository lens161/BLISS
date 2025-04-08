import optuna
from bliss import optimise_bliss
from config import Config
from multiprocessing import freeze_support

def objective(trial):
    bucket_size = trial.suggest_categorical('B', [4096, 8192])
    learning_rate = trial.suggest_categorical('lr', [0.001, 0.005, 0.01])
    batch_size = trial.suggest_categorical('batch_size', [2000, 3000, 4000, 5000])
    m = trial.suggest_categorical('m', [5, 10, 15, 20])

    recall, dist_comps = optimise_bliss(bucket_size, learning_rate, batch_size, m)
    return recall, dist_comps

if __name__ == '__main__':
    study = optuna.create_study(
        storage="sqlite:///opt_bliss.db",
        study_name="optimise_bliss",
        directions=['maximize', 'minimize'],
        load_if_exists=True
    )
    study.optimize(objective, n_trials=20)