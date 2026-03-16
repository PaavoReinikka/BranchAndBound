# Branch and Bound Framework

This sub-crate provides a generalized, high-performance **Branch and Bound (B&B)** engine. It decouples the problem-specific logic (states, branching, and bounding) from the search strategies.

## Core Concepts

To use this framework, you need to implement two primary traits: `SearchState` and `SearchProblem`.

### 1. `SearchState`
Represents a single node in the search tree.

```rust
pub trait SearchState: Send + Sync + Clone {
    type Key: std::hash::Hash + Eq + Send + Sync + Clone;
    
    /// A unique identifier for the state to prevent redundant searches or cycles.
    fn key(&self) -> Self::Key;
    
    /// The current depth of the node in the search tree.
    fn depth(&self) -> usize;
}
```

### 2. `SearchProblem`
Defines the rules, evaluation, and pruning logic for your specific optimization problem.

```rust
pub trait SearchProblem<S: SearchState, V: ObjectiveValue> {
    /// Returns the initial nodes to start the search.
    fn root_states(&self) -> Vec<S>;
    
    /// Generates the children (descendants) of a given state.
    fn branch(&self, state: &S) -> Vec<S>;
    
    /// Calculates the actual objective value of a state (if it is a valid solution).
    fn evaluate(&self, state: &S) -> Option<V>;
    
    /// Calculates an optimistic bound for all descendants of this state.
    /// - For Maximize: The maximum possible value any descendant can reach.
    /// - For Minimize: The minimum possible value any descendant can reach.
    fn bound(&self, state: &S) -> V;

    /// Specifies if we are trying to Maximize or Minimize the objective.
    fn goal(&self) -> OptimizationGoal;
}
```

## Solvers

The framework provides three plug-and-playable solvers in `BranchAndBound::solvers`:

| Solver | Characteristics | Best For... | Parallelism |
| :--- | :--- | :--- | :--- |
| **`BfsSolver`** | Level-by-level traversal. | Shallow trees, broad pruning. | **High** (via `rayon`) |
| **`DfsSolver`** | Deep-first stack-based search. | Memory efficiency. | Low (Single-threaded) |
| **`BestFirstSolver`** | Priority-queue based on `bound()`. | Finding optimal solutions fast. | Low (Priority-queue bound) |

### Basic Usage Example

```rust
use BranchAndBound::{SearchProblem, SearchState, OptimizationGoal};
use BranchAndBound::solvers::BfsSolver;

// 1. Define your state and problem
// ... (Implementation of SearchState and SearchProblem)

// 2. Choose a solver and run the search
let k = 10; // Return Top-10 results
let initial_threshold = 0.0; 
let results = BfsSolver::search(&my_problem, k, initial_threshold);

for node in results {
    println!("Value: {:?}, State: {:?}", node.value, node.state);
}
```

## Key Features

- **Top-K Results:** All solvers track the best `k` solutions found so far and use them to dynamically update pruning thresholds.
- **Generic Optimization:** Supports both `Maximize` and `Minimize` goals through a unified interface.
- **Zero-Cost Abstractions:** Uses Rust generics and traits to ensure that the abstraction layer doesn't introduce runtime overhead.
- **Parallel BFS:** The `BfsSolver` uses `rayon` to explore levels in parallel, making it highly efficient for multi-core systems.

## Requirements

To include this in your project, ensure your `Cargo.toml` includes:
- `rayon` (for parallel BFS)
- `dashmap` (for concurrent level management)
- `parking_lot` (for thread-safe result collection)
