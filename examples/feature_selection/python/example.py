import numpy as np
from feature_selection_bnb import BranchAndBoundSelector
from sklearn.datasets import make_regression

def test_selector():
    print("Testing BranchAndBoundSelector (sklearn-compatible interface)")
    print("=============================================================")
    
    # 1. Create synthetic data
    # X has 10 features, but only 3 are truly informative
    X, y = make_regression(n_samples=100, n_features=10, n_informative=3, noise=0.1, random_state=42)
    
    # 2. Initialize and fit the selector
    selector = BranchAndBoundSelector(metric='bic', max_features=5)
    selector.fit(X, y)
    
    print(f"Selected indices: {selector.selected_indices_}")
    
    # 3. Transform data
    X_new = selector.transform(X)
    print(f"Original shape: {X.shape}")
    print(f"New shape:      {X_new.shape}")
    
    # 4. Check feature names
    print(f"Feature names out: {selector.get_feature_names_out()}")
    
    # 5. Verify consistency
    assert X_new.shape[1] == len(selector.selected_indices_)
    print("\nTest passed successfully!")

if __name__ == "__main__":
    test_selector()
