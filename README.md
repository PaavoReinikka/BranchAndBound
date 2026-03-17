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
- **Maturin:** `pip install maturin` (for Python extension development)

### Build the Project

To build the Rust core and prepare the Python extensions:

```bash
# 1. Build all Rust components
cargo build --release

# 2. Setup Python environment
uv sync

# 3. Build and install Python extensions into the local environment
# This is required for the Python examples to work correctly.
cd kingfisher && uv run maturin develop && cd ..
cd feature_selection && uv run maturin develop && cd ..
```

---

## Examples

All examples have been consolidated into the root `examples/` directory.

### 1. Kingfisher (Rule Mining)

**Rust CLI:**
```bash
cargo run -p kingfisher_bnb -- --data data/test_data.txt --cols 4 --t-type 3
```

**Python Example:**
```bash
uv run python examples/kingfisher/python/example.py
```

### 2. Feature Selection

**Rust Example:**
```bash
cargo run --example simple_regression -p feature_selection
```

**Python Example:**
```bash
uv run python examples/feature_selection/python/example.py
```

### 3. Branch and Bound Core

**Rust Examples:**
```bash
cargo run --example association_rules -p branch_and_bound
cargo run --example association_rules_standalone -p branch_and_bound
```

---

## Usage

### 1. Kingfisher (Rule Mining)
...
### 2. Feature Selection
...
---

## Testing
Run all unit and integration tests across the workspace:
```bash
cargo test
```

## Project Structure

```text
examples/
├── kingfisher/
│   ├── python/          # Python examples for Kingfisher
│   └── rust/            # Rust tests/examples for Kingfisher
├── feature_selection/
│   ├── python/          # Python examples for Feature Selection
│   └── rust/            # Rust examples for Feature Selection
└── branch_and_bound/
    └── rust/            # Core B&B algorithm examples
```

## Design Goals
- **Zero-cost Abstractions:** Use Rust generics to ensure no runtime overhead for the trait system.
- **Parallelism:** Leverage `rayon` for massive scale search spaces (BFS solver).
- **Flexibility:** Optimal **Best-First Search** for objective-driven pruning.
