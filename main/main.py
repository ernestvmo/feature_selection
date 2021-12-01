import os
import collections
from time import time
from typing import Optional
from multiprocessing import Pool

import matplotlib.pyplot as plt
import numpy as np
import openml
import sklearn
import pandas as pd

from skopt import BayesSearchCV

from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel

from functions import get_model_grid, get_datasets, checkpoint


CV_BAYESIAN = 3
CV_EVAL = 10
SCORING = 'roc_auc_ovr_weighted'
N_JOBS = -1
PROCESSES = 64
RANDOM_STATE = 0xBAD
TRAIN_SPLIT = 0.72
BAYESIAN_ITER = 50

MODELS = (
    LogisticRegression(), # Ivan
    NearestNeighbors(), # Koen
    GaussianNB(), # Ernest
    DecisionTreeClassifier(), # Ivan
    RandomForestClassifier(), # Koen
    MLPClassifier(), # Ernest
    GradientBoostingClassifier(), # Koen
    SVC(), # Ernest
)

# DATASETS = get_datasets(random_cnt=2)


def task_generator(datasets=None, models=None):
    if models is None:
        models = MODELS
    
    if datasets is None:
        datasets = DATASETS
    
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
    cv_score_before = cross_val_score(opt, X, y, cv=CV_EVAL)

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

    cv_score_after = cross_val_score(opt, X, y, cv=CV_EVAL)

    res = {"ID": int(dataset_id), "model": model.__class__.__name__,
           "cv_before": cv_score_before.tolist(), "cv_after": cv_score_after.tolist(),
           "time": time()-start_time}

    checkpoint(res, path, rewrite)
    if verbose > 1:
        print('Saved\n\n')

    return res


def dd(model, dataset, path, rewrite=False, verbose=1):
    print(model, dataset.shape, path, rewrite, verbose)
    return {'1':path}


if __name__ == '__main__':
    REWRITE = False
    results_folder = '../../results/'

    gen = list(task_generator_v2('../../datasets'))

    models = []
    datasets = []

    # for i in gen:
    #     models.append(i[0])
    #     datasets.append(i[1])

    get_dataset_model_score(gen[0], results_folder, REWRITE, 5)
    from itertools import repeat
    # with Pool(PROCESSES) as p:
    #     p.starmap(get_dataset_model_score, zip(models, datasets, repeat(results_folder), repeat(REWRITE), repeat(5)))
    # for m, d in gen:
    #     get_dataset_model_score(m, d, results_folder, REWRITE, verbose=5)
