import os
import collections
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import openml
import sklearn
import pandas as pd

from skopt import BayesSearchCV

from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel

from functions import get_model_grid, get_datasets, checkpoint


CV = 3
SCORING = 'roc_auc_ovr_weighted'
N_JOBS = 6

MODELS = (
    SVC(),
    RandomForestClassifier(),
)

DATASETS = get_datasets(random_cnt=2)


def task_generator(datasets=None, models=None):
    if models is None:
        models = MODELS
    
    if datasets is None:
        datasets = DATASETS
    
    for d in datasets:
        for m in models:
            yield m, d
            

def get_dataset_model_score(model, dataset, path, rewrite=False, verbose=1) -> Optional[dict]:
    X, y, dataset_id = dataset.drop(["target", "dataset_id"], axis=1), dataset["target"], dataset["dataset_id"][0]
    if verbose > 0:
        print(f'Starting processing dataset #{dataset_id} with {model.__class__.__name__}')

    if not rewrite:
        file_name = os.path.join(results_folder, f'res-{dataset_id}-{model.__class__.__name__}') + '.json'
        if os.path.exists(file_name):
            return

    hyperparameters = get_model_grid(model)
    if verbose > 1:
        print('Starting regular Bayesian optimization')
    opt = BayesSearchCV(model, hyperparameters, n_iter=2, n_points=5, n_jobs=N_JOBS, cv=3, verbose=1) # Takes a long time!

    cv_score_before = cross_val_score(opt, X, y, cv=CV, n_jobs=N_JOBS)

    # some random feature selection method
    fs = ExtraTreesClassifier(n_estimators=50)
    fs.fit(X, y)
    fsModel = SelectFromModel(fs, prefit=True)
    X = fsModel.transform(X)

    if verbose > 1:
        print('Starting feature selected Bayesian optimization')
    cv_score_after = cross_val_score(opt, X, y, cv=CV, n_jobs=N_JOBS, verbose=1) #scoring=SCORING

    res = {"ID": dataset_id, "model": model.__class__.__name__,
            "cv_before": cv_score_before, "cv_after": cv_score_after}

    checkpoint(res, path, rewrite)
    if verbose > 1:
        print('Saved\n\n')


if __name__ == '__main__':
    REWRITE = False
    results_folder = '../../results/'

    gen = task_generator(get_datasets([11, 15]))

    for m, d in gen:
        get_dataset_model_score(m, d, results_folder, REWRITE, verbose=5)
