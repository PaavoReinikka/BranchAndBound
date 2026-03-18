use branch_and_bound::{SearchProblem, SearchState, OptimizationGoal, solvers::BfsSolver};
use nalgebra::{DMatrix, DVector};
use pyo3::prelude::*;

/// The metric used to evaluate a feature subset.
#[derive(Debug, Clone, Copy)]
pub enum FeatureMetric {
    AIC,
    BIC,
    R2,
    AdjustedR2,
    RSS,
}

impl FeatureMetric {
    fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "aic" => Some(FeatureMetric::AIC),
            "bic" => Some(FeatureMetric::BIC),
            "r2" => Some(FeatureMetric::R2),
            "adjusted_r2" => Some(FeatureMetric::AdjustedR2),
            "rss" => Some(FeatureMetric::RSS),
            _ => None,
        }
    }
}

#[pyfunction]
#[pyo3(signature = (x, y, metric="bic", max_features=5, lambda=0.0))]
fn find_best_features(
    x: Vec<Vec<f64>>,
    y: Vec<f64>,
    metric: &str,
    max_features: usize,
    lambda: f64,
) -> PyResult<Vec<usize>> {
    let n_rows = x.len();
    if n_rows == 0 { return Ok(vec![]); }
    let n_cols = x[0].len();
    
    // Convert Vec<Vec> to DMatrix
    let mut x_mat = DMatrix::zeros(n_rows, n_cols);
    for i in 0..n_rows {
        for j in 0..n_cols {
            x_mat[(i, j)] = x[i][j];
        }
    }
    let y_vec = DVector::from_vec(y);

    let m = FeatureMetric::from_str(metric)
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("Invalid metric"))?;

    let problem = FeatureProblem::new(x_mat, y_vec, m, max_features, lambda);
    
    // Set appropriate initial threshold based on the optimization goal
    let initial_threshold = match problem.goal() {
        OptimizationGoal::Minimize => f64::INFINITY,
        OptimizationGoal::Maximize => f64::NEG_INFINITY,
    };
    
    // We just want the single best one for the sklearn interface
    let results = BfsSolver::search(&problem, 1, initial_threshold);
    
    if let Some(best) = results.first() {
        Ok(best.state.active_indices.clone())
    } else {
        Ok(vec![])
    }
}

#[pymodule]
fn feature_selection_extension(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(find_best_features, m)?)?;
    Ok(())
}

#[derive(Clone, Debug)]
pub struct FeatureState {
    pub active_indices: Vec<usize>,
}

impl SearchState for FeatureState {
    type Key = Vec<usize>;
    fn key(&self) -> Self::Key { self.active_indices.clone() }
    fn depth(&self) -> usize { self.active_indices.len() }
}

pub struct FeatureProblem {
    pub x: DMatrix<f64>,
    pub y: DVector<f64>,
    pub metric: FeatureMetric,
    pub max_features: usize,
    pub lambda: f64, // Regularization parameter
    full_model_rss: f64,
    y_var: f64,
}

impl FeatureProblem {
    pub fn new(x: DMatrix<f64>, y: DVector<f64>, metric: FeatureMetric, max_features: usize, lambda: f64) -> Self {
        let y_mean = y.mean();
        let y_var = y.iter().map(|&yi| (yi - y_mean).powi(2)).sum::<f64>();
        
        // Compute full model RSS (OLS) as an absolute lower bound for pruning
        let (full_model_rss, _) = Self::compute_fit(&x, &y, 0.0);

        Self { x, y, metric, max_features, lambda, full_model_rss, y_var }
    }

    /// Returns (RSS, Effective Degrees of Freedom)
    fn compute_fit(x: &DMatrix<f64>, y: &DVector<f64>, lambda: f64) -> (f64, f64) {
        let k = x.ncols();
        if k == 0 { return (y.dot(y), 0.0); }
        
        let xt = x.transpose();
        let mut xtx = &xt * x;
        let xty = &xt * y;

        // Ridge Regularization: (X^T X + lambda*I)
        if lambda > 0.0 {
            for i in 0..k {
                xtx[(i, i)] += lambda;
            }
        }
        
        match xtx.cholesky() {
            Some(cholesky) => {
                let beta = cholesky.solve(&xty);
                let resid = y - (x * beta);
                let rss = resid.dot(&resid);
                
                // Effective Degrees of Freedom: Tr(X * (X^T X + lambda*I)^-1 * X^T)
                // Which is Tr((X^T X + lambda*I)^-1 * X^T X)
                let df = if lambda > 0.0 {
                    // Re-calculate OLS xtx for the trace part
                    let xtx_ols = &xt * x;
                    let inv_xtx_lambda = cholesky.inverse();
                    (&inv_xtx_lambda * &xtx_ols).trace()
                } else {
                    k as f64
                };

                (rss, df)
            },
            None => (f64::INFINITY, k as f64)
        }
    }
}

impl SearchProblem<FeatureState, f64> for FeatureProblem {
    fn root_states(&self) -> Vec<FeatureState> {
        (0..self.x.ncols()).map(|i| FeatureState { active_indices: vec![i] }).collect()
    }

    fn branch(&self, state: &FeatureState) -> Vec<FeatureState> {
        if state.depth() >= self.max_features { return vec![]; }
        let last = *state.active_indices.last().unwrap_or(&0);
        (last + 1..self.x.ncols()).map(|i| {
            let mut new_indices = state.active_indices.clone();
            new_indices.push(i);
            FeatureState { active_indices: new_indices }
        }).collect()
    }
fn evaluate(&self, state: &FeatureState) -> Option<f64> {
    let n = self.y.len() as f64;
    let sub_x = self.x.select_columns(&state.active_indices);
    let (rss, df) = Self::compute_fit(&sub_x, &self.y, self.lambda);

    if rss == f64::INFINITY { return None; }

    match self.metric {
        FeatureMetric::RSS => Some(rss),
        FeatureMetric::AIC => Some(n * (rss / n).ln() + 2.0 * (df + 1.0)),
        FeatureMetric::BIC => Some(n * (rss / n).ln() + (df + 1.0) * (n.ln())),
        FeatureMetric::R2 => Some(1.0 - (rss / self.y_var)),
        FeatureMetric::AdjustedR2 => {
            let k = state.active_indices.len() as f64;
            Some(1.0 - (rss / (n - k - 1.0)) / (self.y_var / (n - 1.0)))
        },
    }
}

fn bound(&self, _state: &FeatureState) -> f64 {
    let _n = self.y.len() as f64;

    match self.metric {
        FeatureMetric::RSS => self.full_model_rss,
        FeatureMetric::AIC | FeatureMetric::BIC => f64::NEG_INFINITY,
        FeatureMetric::R2 => 1.0,
        FeatureMetric::AdjustedR2 => 1.0,
    }

}
    fn goal(&self) -> OptimizationGoal {
        match self.metric {
            FeatureMetric::R2 | FeatureMetric::AdjustedR2 => OptimizationGoal::Maximize,
            _ => OptimizationGoal::Minimize,
        }
    }
}
