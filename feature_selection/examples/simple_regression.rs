use feature_selection_extension::{FeatureProblem, FeatureMetric};
use branch_and_bound::solvers::BfsSolver;
use nalgebra::{DMatrix, DVector};

fn main() {
    let n = 100;
    let m = 10;
    let mut x_data = DMatrix::zeros(n, m);
    let mut y_data = DVector::zeros(n);
    
    for i in 0..n {
        for j in 0..m {
            x_data[(i, j)] = (((i + 1) * (j + 13)) % 19) as f64 * 0.1;
        }
        let x0 = x_data[(i, 0)];
        let x3 = x_data[(i, 3)];
        y_data[i] = 5.0 * x0 - 2.0 * x3 + ((i % 7) as f64 * 0.001);
    }

    println!("Comparison of Feature Selection Metrics");
    println!("=======================================");

    // 1. Run with R-Squared (Expect it to just fill up to max_features)
    let problem_r2 = FeatureProblem::new(x_data.clone(), y_data.clone(), FeatureMetric::R2, 4, 0.0);
    let results_r2 = BfsSolver::search(&problem_r2, 3, f64::NEG_INFINITY);
    
    println!("\nTop 3 with R-Squared (Maximize):");
    for res in results_r2 {
        println!("  Features: {:?}, R2: {:.6}", res.state.active_indices, res.value);
    }

    // 2. Run with Adjusted R-Squared (Expect it to find the "true" model [0, 3])
    let problem_adj = FeatureProblem::new(x_data, y_data, FeatureMetric::AdjustedR2, 4, 0.0);
    let results_adj = BfsSolver::search(&problem_adj, 3, f64::NEG_INFINITY);
    
    println!("\nTop 3 with Adjusted R-Squared (Maximize):");
    for res in results_adj {
        println!("  Features: {:?}, Adj R2: {:.6}", res.state.active_indices, res.value);
    }
}
