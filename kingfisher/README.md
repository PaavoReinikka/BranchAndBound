# Kingfisher Rule Mining (Optimized Reproduction)

This sub-crate provides a high-performance reproduction of the **Kingfisher** algorithm for finding the Top-K statistically significant dependency rules. The original Kingfisher was developed by **Wilhelmiina Hämäläinen** in C. This version provides a modernized Rust core with a simple Python interface.

This reproduction uses the `BranchAndBound` core with a **Best-First Search** strategy, which is the optimal traversal order described in the original papers.

## Features

- **Optimal Search:** Uses a Priority Queue driven by optimistic bounds on the selected statistical measure (Fisher's p, Chi-squared, etc.).
- **Multiple Measures:** Support for Fisher's Exact Test, Chi-squared, Mutual Information, and Leverage.
- **Both Rule Types:** Supports finding both **Positive** (IF A THEN B) and **Negative** (IF A THEN NOT B) dependency rules.
- **Significance Filtering:** Automatically prunes redundant rules that are not more significant than all their parents.
- **Dual Interface:** Full Command Line (CLI) and Python bindings.

## CLI Usage

Run the tool using `cargo`:

```bash
cargo run -p kingfisher_bnb -- [ARGS]
```

### Parameters:
- `-d, --data <PATH>`: Path to transaction data (space-separated indices). Default: `data/test_data.txt`.
- `-c, --cols <NUM>`: Number of attributes in the data. Default: `4`.
- `--top-k <NUM>`: Number of top rules to find. Default: `10`.
- `-m, --max-len <NUM>`: Maximum number of items in a rule (antecedent + consequent). Default: `3`.
- `-r, --t-type <1|2|3>`: Rule type (1: Positive, 2: Negative, 3: Both). Default: `3`.
- `-a, --alpha <FLOAT>`: Significance threshold (e.g., 0.05).
- `-w, --measure <1|3|4|5>`: Measure type: `1` (Fisher's p), `3` (Chi2), `4` (MI), `5` (Leverage). Default: `1`.

## Python Usage

Install the package via `uv` or `pip`:

```bash
uv sync
```

### Example:
```python
import kingfisher

# 1. Start with dense data
dense_data = [[1, 1, 0], [1, 0, 1], [0, 1, 1]]
sparse_data, id_to_name, _ = kingfisher.dense_to_sparse(dense_data, ["Apple", "Banana", "Cherry"])

# 2. Find rules
rules = kingfisher.find_rules_from_data(
    data=sparse_data,
    k=2,        # Max attribute index
    q=10,       # Top-K
    l_max=3,    # Max length
    t_type=3    # Both Pos and Neg
)

# 3. Print with names
for r in rules:
    ant = [id_to_name[i] for i in r.antecedent]
    cons = id_to_name[r.consequent]
    sign = "" if not r.is_negative else "NOT "
    print(f"IF {ant} THEN {sign}{cons} (p={r.measure_value:.4f})")
```

## Performance Note

Unlike a naive level-wise (BFS) version, which is simpler to parallelize, the **Best-First** implementation finds the absolute best rules first. This allows it to set very aggressive pruning thresholds dynamically, often visiting significantly fewer nodes to reach the same result.

*For shallow rules, the BFS strategy can be still very effective.*

## Attribution

Hämäläinen, W.: *Efficient discovery of the top-K optimal dependency rules with Fisher's exact test of significance*. ICDM 2010.

Original C implementation: [whsivut/kingfisher](https://sites.google.com/site/whsivut/home/sourcecode/kingfisher)

## License

**Important:** Refer to the original author's site for research use and licensing. Although heavily modified, this Rust implementation is still a derivative work.
│
