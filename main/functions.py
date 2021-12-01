import os
import json
import random
from typing import Optional

import pandas as pd
from sklearn.base import is_classifier

PARAMETERS_DIR = '../hyperparameters_grid/'
RESULTS_DIR = '../results/'
DATASETS_DIR = '../../datasets/'


def get_model_grid(model, path=PARAMETERS_DIR) -> Optional[dict]:
    """
    Get a dictionary with hyperparameter grid for a given model
    :param model:
    :param path: path to folder with jsons
    :return: dictionary with parameters
    """
    if not (type(model) == str or is_classifier(model)):
        return None
#         raise Exception('Unknown model')
    
    files = set(file[:-5] for file in os.listdir(path) if file.endswith('.json'))
    model_str = model.lower() if type(model) == str else model.__class__.__name__.lower()
    
    if model_str not in files:
        return None
#         raise Exception('Unknown model')
    
    with open(os.path.join(path, f'{model_str}.json')) as f:
        hyperparameter_grid = json.load(f)
        
    return hyperparameter_grid


# {"ID": dataset_id, "model": model.__class__.__name__, "cv_before": np.mean(cv_score_before),
# "cv_after": np.mean(cv_score_after)}
def checkpoint(row: dict, path=RESULTS_DIR, rewrite=True):
    if not os.path.exists(path):
        os.mkdir(path)

    file_name = os.path.join(path, f'res-{row.get("ID")}-{row.get("model")}') + '.json'

    if not rewrite and os.path.exists(file_name):
        return

    with open(file_name, 'w') as f:
        json.dump(row, f, indent=1)


def get_datasets(ids=None, random_cnt: int = 0, path=DATASETS_DIR):
    """
    Creates a generator of datasets, can work in ids, random and combined mode.
    :param ids: select ids
    :param random_cnt: number of randomly picked datasets
    :param path: path to dataset folder
    :return: generator of datasets
    """
    all_ids = tuple(file[3:-7] for file in os.listdir(path) if file.endswith('.csv.gz'))
    
    if ids is None:
        ids = []

    ids = [str(i) for i in ids]
    
    if random:
        ids += random.choices(all_ids, k=random_cnt)

    for dataset_id in set(ids):
        if dataset_id not in all_ids:
            continue

        df = pd.read_csv(os.path.join(path, f'df-{dataset_id}.csv.gz'))
        yield df


def load_datasets(path=DATASETS_DIR):
    import openml
    benchmark_suite = openml.study.get_suite(suite_id='OpenML-CC18')

    datasets = []
    print(benchmark_suite, "\n======================")
    for dataset_id in benchmark_suite.data:
        data = openml.datasets.get_dataset(dataset_id)
        if len(data.features) < 2000:  # This skips one huge dataset, but reduces time
            X, y, categorical_indicator, attribute_names = data.get_data(
                dataset_format="array", target=data.default_target_attribute
            )
            df = pd.DataFrame(X, columns=attribute_names)
            df["target"] = y
            df["dataset_id"] = data.id
            datasets.append(df)

    # print(f"{len(datasets)}/{len(benchmark_suite.data)} datasets used")

    for dataset in datasets:
        dataset_id = dataset.dataset_id[0]
        dataset.to_csv(os.path.join(path, f'df-{dataset_id}.csv.gz'), index=False, compression='gzip')
