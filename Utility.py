import os
import sys
import logging
import yaml

import pandas as pd

def load_data(data_path, labels, X_path, y_path, data_has_index):
    if data_path:
        index_col = 0 if data_has_index else None
        data = pd.read_csv(data_path, index_col=index_col)
        X = data.drop(labels, axis=1)
        y = data[labels]
    else:
        index_col = 0 if data_has_index else None
        X = pd.read_csv(X_path, index_col=index_col)
        y = pd.read_csv(y_path, index_col=index_col)
    return X, y

def ensure_exists_dir(path):
    if not os.path.exists(path):
        # Note: No logging allowed here
        os.makedirs(path)
        return True
    return False

def load_config(config_path):
    with open(config_path, "r") as f:
        # TODO: Look into Unicode loading to cover all paths (e.g. with umlauts)
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config

def init_logging(logging_level, results_path):
    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging_level,
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            # Log to both file and console (does not affect Optuna logging, so you only get the "full" output in the console)
            logging.FileHandler(os.path.join(results_path, f"log.txt")),
            logging.StreamHandler(sys.stdout)
        ]
    )