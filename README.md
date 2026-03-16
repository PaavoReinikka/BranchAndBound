# Generalized Branch and Bound Framework (B&B)

This framework provides a generalized implementation of the Branch and Bound optimization paradigm. It is designed to be extensible, allowing developers to plug in custom search problems while leveraging high-performance, parallel search strategies.

## Core Projects

| Directory | Crate | Description |
| :--- | :--- | :--- |
| `branch_and_bound/` | `branch_and_bound` | The core engine providing BFS, DFS, and Best-First solvers. |
| `kingfisher/` | `kingfisher_bnb` | Top-K non-redundant dependency rule mining using Fisher's Exact Test. |
| `feature_selection/` | `feature_selection` | Parallel feature selection for linear regression (RSS, AIC, BIC, R2). |

---

## Installation & Build

### Prerequisites
- **Rust:** `cargo` and `rustc` (Edition 2021)
- **Python:** 3.8+ and `uv` (recommended)

### Clean Build
To clean and rebuild the entire workspace (Rust core + Python extensions):

```bash
# Clean previous artifacts
cargo clean

# Build all Rust components
cargo build --release

# Setup Python environment and build extensions
uv sync
```

---

## Usage

### 1. Kingfisher (Rule Mining)

**CLI:**
```bash
cargo run -p kingfisher_bnb -- --data data/test_data.txt --cols 4 --t-type 3
```

**Python:**
```python
import kingfisher_bnb
rules = kingfisher_bnb.find_rules_from_data(data=my_sparse_data, k=10, q=100)
```

### 2. Feature Selection

**Python (Scikit-Learn Interface):**
```python
from feature_selection_bnb import BranchAndBoundSelector
selector = BranchAndBoundSelector(k_features=5, metric='bic')
selector.fit(X, y)
```

---

## Testing
Run all unit and integration tests across the workspace:
```bash
cargo test
```

## Design Goals
- **Zero-cost Abstractions:** Use Rust generics to ensure no runtime overhead for the trait system.
- **Parallelism:** Leverage `rayon` for massive scale search spaces (BFS solver).
- **Flexibility:** Optimal **Best-First Search** for objective-driven pruning.
