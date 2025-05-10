import numpy as np
import pandas as pd

from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

class BinaryOneClassSVM(OneClassSVM):
    def predict(self, X):
        y_pred = super().predict(X)
        return (y_pred == -1).astype(int) # -1 -> 1, 1 -> 0
    
class BinaryIsolationForest(IsolationForest):
    def predict(self, X):
        y_pred = super().predict(X)
        return (y_pred == -1).astype(int) # -1 -> 1, 1 -> 0
    
class BinaryLocalOutlierFactor(LocalOutlierFactor):
    def predict(self, X):
        y_pred = super().fit_predict(X)
        return (y_pred == -1).astype(int) # -1 -> 1, 1 -> 0
    
def split_with_one_class_KFold(X, y, n_splits, shuffle, random_state):
    # Has to be KFold. StratifiedKFold would cause data leakage because it splits based on the target which is supposed to be one-class
    kfold = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    splits = kfold.split(X) # Iterator of (train_index, valid_index) tuples
    # Generator for performance and maintaining the iterator
    # for train_index, valid_index in splits:
    #     yield (train_index[y.iloc[train_index] == 0], valid_index)
    # Shouldnt be a generator because it cant be pickled
    return [(train_index[y.iloc[train_index] == 0], valid_index) for train_index, valid_index in splits]

class PandasStandardScaler(StandardScaler):
    def transform(self, X):
        X_transformed = super().transform(X)
        return pd.DataFrame(X_transformed, index=X.index, columns=X.columns)
    
    def fit_transform(self, X, y=None):
        X_transformed = super().fit_transform(X, y)
        return pd.DataFrame(X_transformed, index=X.index, columns=X.columns)
