use branch_and_bound::{SearchProblem, SearchState, OptimizationGoal, solvers::{BfsSolver, DfsSolver, BestFirstSolver}, ResultNode};

#[derive(Clone, Debug)]
struct RuleState {
    path: Vec<usize>,
}

impl SearchState for RuleState {
    type Key = Vec<usize>;
    fn key(&self) -> Self::Key { self.path.clone() }
    fn depth(&self) -> usize { self.path.len() }
}

struct RuleProblem {
    num_attrs: usize,
    max_depth: usize,
}

impl SearchProblem<RuleState, f64> for RuleProblem {
    fn root_states(&self) -> Vec<RuleState> {
        (0..self.num_attrs).map(|i| RuleState { path: vec![i] }).collect()
    }

    fn branch(&self, state: &RuleState) -> Vec<RuleState> {
        if state.depth() >= self.max_depth { return vec![]; }
        let last = *state.path.last().unwrap();
        (last + 1..self.num_attrs).map(|i| {
            let mut new_path = state.path.clone();
            new_path.push(i);
            RuleState { path: new_path }
        }).collect()
    }

    fn evaluate(&self, state: &RuleState) -> Option<f64> {
        // Simplified dummy evaluation
        Some(1.0 / (state.path.iter().sum::<usize>() as f64 + 1.0))
    }

    fn bound(&self, _state: &RuleState) -> f64 {
        // In this dummy problem, adding more items (increasing sum) 
        // will always result in a SMALLER value (better for minimization).
        // The best possible value for any descendant is 0.0.
        0.0
    }

    fn goal(&self) -> OptimizationGoal {
        OptimizationGoal::Minimize
    }
}

fn print_results(header: &str, results: &[ResultNode<RuleState, f64>]) {
    println!("\n--- {} ---", header);
    for (i, res) in results.iter().enumerate() {
        println!("{}. Path: {:?}, Value: {:.4}", i + 1, res.state.path, res.value);
    }
}

fn main() {
    let problem = RuleProblem { num_attrs: 10, max_depth: 3 };
    let k = 5;
    let initial_threshold = 1.0;

    let bfs_res = BfsSolver::search(&problem, k, initial_threshold);
    print_results("BFS Solver", &bfs_res);

    let dfs_res = DfsSolver::search(&problem, k, initial_threshold);
    print_results("DFS Solver", &dfs_res);

    let best_res = BestFirstSolver::search(&problem, k, initial_threshold);
    print_results("Best-First Solver", &best_res);
}
