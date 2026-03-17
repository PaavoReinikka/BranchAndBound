use kingfisher_bnb_extension::{KingfisherProblem, BitMatrix, Measures};
use branch_and_bound::solvers::BestFirstSolver;
use std::path::Path;
use std::sync::Arc;
use clap::Parser;

#[derive(Parser, Debug)]
#[command(author, version, about = "Kingfisher Rule Mining (Best-First BnB)")]
struct Args {
    /// Path to the transaction data file
    #[arg(short, long, default_value = "data/test_data.txt")]
    data: String,

    /// Number of columns (attributes) in the data
    #[arg(short, long, default_value_t = 4)]
    cols: usize,

    /// Number of top rules to find (q)
    #[arg(long, default_value_t = 10)]
    top_k: usize,

    /// Maximum rule length (l_max)
    #[arg(short, long, default_value_t = 3)]
    max_len: usize,

    /// Rule type: 1=Positive, 2=Negative, 3=Both
    #[arg(short = 'r', long, default_value_t = 3)]
    t_type: u8,

    /// Significance threshold (alpha)
    #[arg(short, long, default_value_t = 0.05)]
    alpha: f64,
}

fn main() {
    let args = Args::parse();

    if !Path::new(&args.data).exists() {
        eprintln!("Error: Data file not found at {}", args.data);
        return;
    }

    let matrix = BitMatrix::load_from_file(&args.data, args.cols).expect("Failed to load matrix");
    let n = matrix.num_rows;
    let problem = KingfisherProblem::new(
        matrix,
        Measures::new(n),
        args.top_k,
        args.max_len,
        1,    // min_fr
        0.0,  // min_cf
        args.t_type,
        1,    // Fisher's p
    );

    println!("Kingfisher Rule Mining");
    println!("----------------------");
    println!("Data: {}, Rows: {}, Cols: {}", args.data, n, problem.matrix.num_cols);
    println!("Goal: Find Top-{} rules (Length <= {}, Type: {})", args.top_k, args.max_len, 
        match args.t_type { 1 => "Pos", 2 => "Neg", _ => "Both" });
    println!("");

    println!("{:<4} | {:<20} | {:<5} | {:<10} | {:<10}", "Rank", "Antecedent", "Type", "Consequent", "p_ln");
    println!("{:-<4}-|-{:-<20}-|-{:-<5}-|-{:-<10}-|-{:-<10}", "", "", "", "", "");

    // Using Best-First solver
    BestFirstSolver::search(&problem, args.top_k, args.alpha.ln());
    
    // Extract and print rules
    let rules_mutex = Arc::try_unwrap(problem.ruleset)
        .expect("Failed to unwrap Arc")
        .into_inner()
        .expect("Failed to unlock Mutex");
    let rules = rules_mutex.into_sorted_vec();

    for (i, rule) in rules.iter().enumerate() {
        let ant_str = format!("{:?}", rule.antecedent);
        let rule_type = if rule.is_negative { "NEG" } else { "POS" };
        println!("{:<4} | {:<20} | {:<5} | {:<10} | {:<10.4}", 
            i + 1, ant_str, rule_type, rule.consequent, rule.measure_value);
    }
}
