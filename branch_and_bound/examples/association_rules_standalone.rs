use branch_and_bound::{SearchProblem, SearchState, OptimizationGoal, solvers::BfsSolver};
use bitvec::prelude::*;
use std::fs::File;
use std::io::{BufRead, BufReader};

// --- Problem-specific data structures (Association Rules) ---

struct BitMatrix {
    attributes: Vec<BitVec<u64, Lsb0>>,
    num_rows: usize,
    num_cols: usize,
}

impl BitMatrix {
    fn load_from_file(path: &str, num_cols: usize) -> std::io::Result<Self> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let mut attributes = vec![BitVec::repeat(false, 1000); num_cols]; // Simple fixed size for demo
        let mut row_count = 0;

        for line in reader.lines() {
            let line = line?;
            let parts: Vec<usize> = line.split_whitespace()
                .filter_map(|s| s.parse().ok())
                .collect();
            
            for &attr in &parts {
                if attr < num_cols {
                    if row_count >= attributes[attr].len() {
                        attributes[attr].resize(row_count + 1000, false);
                    }
                    attributes[attr].set(row_count, true);
                }
            }
            row_count += 1;
        }

        for attr in 0..num_cols {
            attributes[attr].truncate(row_count);
        }

        Ok(BitMatrix { attributes, num_rows: row_count, num_cols })
    }

    fn frequency(&self, path: &[usize]) -> usize {
        if path.is_empty() { return self.num_rows; }
        let mut res = self.attributes[path[0]].clone();
        for &attr in &path[1..] {
            res &= &self.attributes[attr];
        }
        res.count_ones()
    }
}

struct Measures {
    ln_factorials: Vec<f64>,
}

impl Measures {
    fn new(n: usize) -> Self {
        let mut ln_factorials = Vec::with_capacity(n + 1);
        let mut sum = 0.0;
        ln_factorials.push(0.0);
        for i in 1..=n {
            sum += (i as f64).ln();
            ln_factorials.push(sum);
        }
        Measures { ln_factorials }
    }

    fn ln_combination(&self, n: usize, k: usize) -> f64 {
        if k > n { return f64::NEG_INFINITY; }
        self.ln_factorials[n] - self.ln_factorials[k] - self.ln_factorials[n - k]
    }

    fn ln_fishers_p(&self, fr_xa: usize, fr_x: usize, fr_a: usize, n: usize) -> f64 {
        let log_denom = self.ln_combination(n, fr_x);
        let min_xa = fr_xa;
        let max_xa = fr_x.min(fr_a);
        
        let mut terms = Vec::new();
        let mut max_log_p = f64::NEG_INFINITY;
        for i in min_xa..=max_xa {
            let log_p = self.ln_combination(fr_a, i) + self.ln_combination(n - fr_a, fr_x - i) - log_denom;
            if log_p > max_log_p { max_log_p = log_p; }
            terms.push(log_p);
        }
        
        let sum_p: f64 = terms.iter().map(|&p| (p - max_log_p).exp()).sum();
        max_log_p + sum_p.ln()
    }

    fn bound(&self, fr_x: usize, fr_a: usize, n: usize) -> f64 {
        if fr_x <= fr_a {
            self.ln_combination(fr_a, fr_x) - self.ln_combination(n, fr_x)
        } else {
            self.ln_combination(fr_x, fr_a) - self.ln_combination(n, fr_a)
        }
    }
}

// --- SearchState implementation ---

#[derive(Clone, Debug)]
struct AssociationRuleState {
    path: Vec<usize>,
    freq: usize,
}

impl SearchState for AssociationRuleState {
    type Key = Vec<usize>;
    fn key(&self) -> Self::Key { self.path.clone() }
    fn depth(&self) -> usize { self.path.len() }
}

// --- SearchProblem implementation ---

struct AssociationRuleProblem {
    matrix: BitMatrix,
    measures: Measures,
    l_max: usize,
    min_fr: usize,
    min_cf: f64,
}

impl SearchProblem<AssociationRuleState, f64> for AssociationRuleProblem {
    fn root_states(&self) -> Vec<AssociationRuleState> {
        (0..self.matrix.num_cols)
            .map(|i| {
                let freq = self.matrix.attributes[i].count_ones();
                AssociationRuleState { path: vec![i], freq }
            })
            .filter(|s| s.freq >= self.min_fr)
            .collect()
    }

    fn branch(&self, state: &AssociationRuleState) -> Vec<AssociationRuleState> {
        if state.depth() >= self.l_max { return vec![]; }
        let last = *state.path.last().unwrap();
        (last + 1..self.matrix.num_cols)
            .map(|i| {
                let mut new_path = state.path.clone();
                new_path.push(i);
                let freq = self.matrix.frequency(&new_path);
                AssociationRuleState { path: new_path, freq }
            })
            .filter(|s| s.freq >= self.min_fr)
            .collect()
    }

    fn evaluate(&self, state: &AssociationRuleState) -> Option<f64> {
        let mut best_p = f64::INFINITY;
        let n = self.matrix.num_rows;
        
        for &consequent in &state.path {
            let mut antecedent = state.path.clone();
            antecedent.retain(|&x| x != consequent);
            if antecedent.is_empty() { continue; }
            
            let freq_x_ant = self.matrix.frequency(&antecedent);
            let freq_xa = state.freq;
            let freq_a = self.matrix.attributes[consequent].count_ones();
            
            if freq_xa as f64 / freq_x_ant as f64 >= self.min_cf {
                let p = self.measures.ln_fishers_p(freq_xa, freq_x_ant, freq_a, n);
                if p < best_p { best_p = p; }
            }
        }
        if best_p == f64::INFINITY { None } else { Some(best_p) }
    }

    fn bound(&self, state: &AssociationRuleState) -> f64 {
        let n = self.matrix.num_rows;
        let mut best_possible = f64::INFINITY;
        for i in 0..self.matrix.num_cols {
            if !state.path.contains(&i) {
                let freq_a = self.matrix.attributes[i].count_ones();
                let b = self.measures.bound(state.freq, freq_a, n);
                if b < best_possible { best_possible = b; }
            }
        }
        best_possible
    }

    fn goal(&self) -> OptimizationGoal {
        OptimizationGoal::Minimize
    }
}

fn main() {
    // Try to load test data from the original kingfisher path
    let matrix = BitMatrix::load_from_file("../data/test_data.txt", 4).unwrap();
    let n = matrix.num_rows;
    let problem = AssociationRuleProblem {
        matrix,
        measures: Measures::new(n),
        l_max: 3,
        min_fr: 1,
        min_cf: 0.0,
    };

    println!("Running Standalone BFS Search (No kingfisher_rust dependency)...");
    let results = BfsSolver::search(&problem, 10, 0.05f64.ln());
    
    for (i, res) in results.iter().enumerate() {
        println!("{}. Path: {:?}, Best p-value (ln): {:.4}", i + 1, res.state.path, res.value);
    }
}
