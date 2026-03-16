# Generalized Branch and Bound Framework (B&B)

This framework provides a generalized implementation of the Branch and Bound optimization paradigm, inspired by the Kingfisher association rule mining algorithm. It is designed to be extensible for various combinatorial optimization problems, including graph problems, feature selection, and rule mining.

## Architecture

The framework decouples the **problem domain** (what is being searched) from the **search strategy** (how the search tree is traversed).

### 1. Core Abstractions

#### `SearchState` (Trait)
Represents a node in the search tree.
- **`path()`**: Unique identifier or path to this node.
- **`is_valid()`**: Whether this state represents a potential solution.
- **`depth()`**: Current level in the search tree.

#### `SearchProblem<S: SearchState>` (Trait)
Defines the rules of the problem.
- **`root_states()`**: The initial nodes to start the search.
- **`branch(state: &S)`**: Generates the next possible states (children).
- **`evaluate(state: &S)`**: Calculates the actual objective value of a state.
- **`bound(state: &S)`**: Calculates the *optimistic* bound (the best possible value any descendant could achieve).
- **`is_better(v1: Value, v2: Value)`**: Defines if we are maximizing or minimizing.

### 2. Search Strategies

The framework will support three primary traversal methods:

| Strategy | Data Structure | Best For... | Parallelization |
| :--- | :--- | :--- | :--- |
| **BFS (Level-wise)** | `Vec<Vec<S>>` | Finding small solutions, broad pruning | High (Level-by-level) |
| **DFS** | `Stack` | Memory efficiency, deep trees | Low (Single branch) |
| **Best-First** | `PriorityQueue` | Finding optimal solutions fast | Medium (Node-by-node) |

### 3. Pruning Mechanism

Pruning occurs in two ways:
1.  **Local Pruning:** If `bound(state)` is worse than a user-defined threshold, the entire branch is discarded.
2.  **Global Pruning:** If `bound(state)` is worse than the current *worst* value in the top-K results, the branch is discarded.

---

## Roadmap

### Phase 1: Foundation (Complete)
- [x] Define core traits (`SearchState`, `SearchProblem`, `Solver`).
- [x] Implement `BfsSolver` (parallelized via `rayon`).
- [x] Implement `TopK` result collector to manage pruning thresholds.

### Phase 2: First Use Case (Complete)
- [x] Port Kingfisher's rule mining logic into the `SearchProblem` trait.
- [] Verify consistency with the original `kingfisher_rust` implementation.

### Phase 3: Strategy Expansion (Complete)
- [x] Implement `DfsSolver` for memory-constrained searches.
- [x] Implement `BestFirstSolver` for objective-driven optimization.

### Phase 4: Machine Learning Applications (Partial)
- [] **Feature Selection:** Implemented OLS/Ridge with AIC, BIC, R2, and RSS metrics.
- [ ] **Graph Problems:** Implement Max-Clique or Shortest Path as a `SearchProblem`.

## Design Goals
- **Zero-cost Abstractions:** Use Rust generics to ensure no runtime overhead for the trait system.
- **Parallelism:** Leverage `rayon` for massive scale search spaces.
- **Safety:** Ensure thread-safe state management during pruning. **TODO:** Check dashmap vs rayon problem.
