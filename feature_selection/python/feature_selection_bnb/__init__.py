import numpy as np
from sklearn.base import BaseEstimator
from sklearn.feature_selection import SelectorMixin
try:
    from .feature_selection_extension import find_best_features
except ImportError:
    def find_best_features(*args, **kwargs):
        raise ImportError("Rust extension 'feature_selection_extension' not found. Please build the project.")

class BranchAndBoundSelector(BaseEstimator, SelectorMixin):
    def __init__(self, metric='bic', max_features=5, lambda_=0.0):
        self.metric = metric
        self.max_features = max_features
        self.lambda_ = lambda_
        self.support_ = None
        self.selected_indices_ = None

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        selected_indices = find_best_features(
            X.tolist(), 
            y.tolist(), 
            self.metric, 
            self.max_features, 
            self.lambda_
        )
        self.selected_indices_ = np.sort(selected_indices)
        self.support_ = np.zeros(X.shape[1], dtype=bool)
        self.support_[self.selected_indices_] = True
        return self

    def _get_support_mask(self):
        if self.support_ is None:
            raise ValueError("Selector must be fitted before use.")
        return self.support_

    def get_feature_names_out(self, input_features=None):
        mask = self._get_support_mask()
        if input_features is None:
            return np.array([f"x{i}" for i in range(len(mask))])[mask]
        return np.asarray(input_features)[mask]
