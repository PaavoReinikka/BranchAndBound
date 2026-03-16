# Feature Selection BnB

A scikit-learn compatible feature selector using a parallelized Branch and Bound engine.

## Usage

```python
from feature_selection_bnb import BranchAndBoundSelector
from sklearn.datasets import make_regression

X, y = make_regression(n_samples=100, n_features=10, n_informative=3)
selector = BranchAndBoundSelector(metric='bic', max_features=3)
selector.fit(X, y)

print(selector.get_feature_names_out())
```
