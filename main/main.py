import os
import traceback
from itertools import repeat
from time import time
from typing import Optional
from multiprocessing import Pool


from skopt import BayesSearchCV

from sklearn.model_selection import train_test_split, cross_validate
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel

from functions import get_model_grid, get_datasets, checkpoint


CV_BAYESIAN = 3
CV_EVAL = 10
SCORING = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted', 'roc_auc_ovr_weighted']
N_JOBS = 1
PROCESSES = 1
RANDOM_STATE = 0xBAD
TRAIN_SPLIT = 0.72
BAYESIAN_ITER = 50
VERBOSE = 2
REWRITE = False
results_folder = '../../results/'
DATASET_FOLDER = '../../datasets/'

MODELS = (
    LogisticRegression(),
    KNeighborsClassifier(),
    GaussianNB(),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    MLPClassifier(),
    GradientBoostingClassifier(),
    SVC()
)


def task_generator(datasets=None, models=None):
    if models is None:
        models = MODELS
    
    if datasets is None:
        datasets = get_datasets(random_cnt=2)
    
    for d in datasets:
        for m in models:
            yield m, d


def task_generator_v2(datasets_path=None, models=None):
    all_ids = sorted([file for file in os.listdir(datasets_path) if file.endswith('.csv.gz')],
                     key=lambda x: os.stat(os.path.join(datasets_path, x)).st_size)

    all_ids = [i[3: -7] for i in all_ids]

    if models is None:
        models = MODELS

    for d in all_ids:
        for m in models:
            yield m, d
            

def get_dataset_model_score(model2dataset_id, path, rewrite=False, verbose=1) -> Optional[dict]:
    try:
        model = model2dataset_id[0]
        dataset = list(get_datasets(ids=[model2dataset_id[1]]))[0]

        start_time = time()
        X, y, dataset_id = dataset.drop(["target", "dataset_id"], axis=1), dataset["target"], dataset["dataset_id"][0]
        if verbose > 0:
            print(f'Starting processing dataset #{dataset_id} with {model.__class__.__name__}')

        if not rewrite:
            file_name = os.path.join(results_folder, f'res-{dataset_id}-{model.__class__.__name__}') + '.json'
            if os.path.exists(file_name):
                return

        hyperparameters = get_model_grid(model)
        if hyperparameters is None:
            return

        if verbose > 1:
            print('Starting regular Bayesian optimization')

        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=TRAIN_SPLIT, random_state=RANDOM_STATE)
        bayesian = BayesSearchCV(model, hyperparameters, n_iter=BAYESIAN_ITER, n_points=1,
                                 n_jobs=N_JOBS, cv=CV_BAYESIAN, random_state=RANDOM_STATE)
        bayesian = bayesian.fit(X_train, y_train)
        opt = bayesian.best_estimator_
        cv_score_before = cross_validate(opt, X, y, cv=CV_EVAL, scoring=SCORING)
        cv_score_before = {k: list(v) for k, v in cv_score_before.items()}

        before_time = time()
        if verbose > 1:
            print('Starting feature selected Bayesian optimization')

        fs_t = ExtraTreesClassifier(n_estimators=100)
        fs_t.fit(X, y)
        fsModel = SelectFromModel(fs_t, prefit=True)
        X = fsModel.transform(X)

        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=TRAIN_SPLIT, random_state=RANDOM_STATE)
        bayesian = BayesSearchCV(model, hyperparameters, n_iter=BAYESIAN_ITER, n_points=1,
                                 n_jobs=N_JOBS, cv=CV_BAYESIAN, random_state=RANDOM_STATE)
        bayesian = bayesian.fit(X_train, y_train)
        opt = bayesian.best_estimator_

        cv_score_after = cross_validate(opt, X, y, cv=CV_EVAL, scoring=SCORING)
        cv_score_after = {k: list(v) for k, v in cv_score_after.items()}

        end_time = time()

        res = {"ID": int(dataset_id), "model": model.__class__.__name__, "cv_before": cv_score_before,
               "cv_after": cv_score_after, 'total_time': end_time - start_time,
               "regular_time": before_time-start_time, 'featured_time': end_time - before_time}

        checkpoint(res, path, rewrite)
        if verbose > 1:
            print('Saved\n\n')

        return res
    except Exception as e:
        with open('errors.log', 'w') as f:
            traceback.print_exc(file=f)


if __name__ == '__main__':
    gen = list(task_generator_v2(DATASET_FOLDER))
    print(f'Starting executing {len(gen)} tasks.\n\n')

    # for i in range(20):
    #     get_dataset_model_score(gen[i], results_folder, True, 5)
    with Pool(PROCESSES) as p:
        p.starmap(get_dataset_model_score, zip(gen, repeat(results_folder), repeat(REWRITE), repeat(VERBOSE)))
