import numpy as np
import pandas as pd

import random

from imblearn.over_sampling.base import BaseOverSampler
from sklearn.neighbors import NearestNeighbors

class MLSMOTE(BaseOverSampler):
    """
    Implementation of MLSMOTE algorithm as an imblearn over-sampling technique based
    on https://medium.com/thecyphy/handling-data-imbalance-in-multi-label-classification-mlsmote-531155416b87
    and https://www.sciencedirect.com/science/article/pii/S0950705115002737.
    Code partly adapted to work with numpy and with the sklearn and imblearn APIs.
    """

    def __init__(
        self,
        *,
        sampling_strategy=0.2
    ):
        super().__init__(
            sampling_strategy=sampling_strategy
        )

    def _apply_MLSMOTE_numpy(self, X, y, n_sample):
        indices2 = self._nearest_neighbor(X)
        n = len(indices2)
        new_X = np.zeros((n_sample, X.shape[1]))
        target = np.zeros((n_sample, y.shape[1]))

        for i in range(n_sample):
            reference = random.randint(0, n-1)
            neighbour = random.choice(indices2[reference, 1:])
            all_point = indices2[reference]
            
            nn_labels = y[all_point]
            ser = nn_labels.sum(axis=0)
            target[i] = np.array([1 if val > 2 else 0 for val in ser])
            
            ratio = random.random()
            gap = X[reference] - X[neighbour]
            new_X[i] = X[reference] + ratio * gap

        new_X = np.vstack((X, new_X))
        target = np.vstack((y, target))

        # Checks
        assert new_X.shape[0] == target.shape[0], f"{new_X.shape[0]} != {target.shape[0]}"
        assert target.shape[1] == y.shape[1], f"{target.shape[1]} != {y.shape[1]}"
        assert new_X.shape[1] == X.shape[1], f"{new_X.shape[1]} != {X.shape[1]}"
        assert new_X.shape[0] == n_sample + X.shape[0], f"{new_X.shape[0]} != {n_sample + X.shape[0]}"
        assert target.shape[0] == n_sample + y.shape[0], f"{target.shape[0]} != {n_sample + y.shape[0]}"

        return new_X, target
    
    def _apply_MLSMOTE_pandas(self, X, y, n_sample):        
        indices2 = self._nearest_neighbor(X)
        n = len(indices2)
        new_X = np.zeros((n_sample, X.shape[1]))
        target = np.zeros((n_sample, y.shape[1]))
        for i in range(n_sample):
            reference = random.randint(0,n-1)
            neighbour = random.choice(indices2[reference,1:])
            all_point = indices2[reference]
            nn_df = y[y.index.isin(all_point)]
            ser = nn_df.sum(axis = 0, skipna = True)
            target[i] = np.array([1 if val>2 else 0 for val in ser])
            ratio = random.random()
            gap = X.loc[reference,:] - X.loc[neighbour,:]
            new_X[i] = np.array(X.loc[reference,:] + ratio * gap)
        new_X = pd.DataFrame(new_X, columns=X.columns)
        target = pd.DataFrame(target, columns=y.columns)
        new_X = pd.concat([X, new_X], axis=0)
        target = pd.concat([y, target], axis=0)

        # Checks
        assert new_X.shape[0] == target.shape[0], f"{new_X.shape[0]} != {target.shape[0]}"
        assert target.shape[1] == y.shape[1], f"{target.shape[1]} != {y.shape[1]}"
        assert new_X.shape[1] == X.shape[1], f"{new_X.shape[1]} != {X.shape[1]}"
        assert new_X.shape[0] == n_sample + X.shape[0], f"{new_X.shape[0]} != {n_sample + X.shape[0]}"
        assert target.shape[0] == n_sample + y.shape[0], f"{target.shape[0]} != {n_sample + y.shape[0]}"

        return new_X, target
    
    def _nearest_neighbor(self, X):
        nbs=NearestNeighbors(n_neighbors=5,metric='euclidean',algorithm='kd_tree').fit(X)
        euclidean,indices= nbs.kneighbors(X)
        return indices
        
    def _fit_resample(self, X, y):
        if 0 < self.sampling_strategy < 1: # A fraction of the original data length
            self.sampling_strategy = int(X.shape[0] * self.sampling_strategy)
        elif self.sampling_strategy > 1 and int(self.sampling_strategy) == self.sampling_strategy: # A number of samples
            self.sampling_strategy = int(self.sampling_strategy)

        if isinstance(X, pd.DataFrame):
            return self._apply_MLSMOTE_pandas(X, y, self.sampling_strategy)
        elif isinstance(X, np.ndarray):
            if not isinstance(y, np.ndarray): # Happens
                y = y.to_numpy()
            return self._apply_MLSMOTE_numpy(X, y, self.sampling_strategy)
    
    # Its important to overwrite this too
    def fit_resample(self, X, y):
        return self._fit_resample(X, y)
    
    def fit(self, X, y):
        return self