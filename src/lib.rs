//! # Branch & Bound Template
//!
//! This crate provides a general template for Branch & Bound algorithms.
//!
//! ## Example
//!
//! This is a rather long example.
//! It models the [Knapsack-Problem](https://en.wikipedia.org/wiki/Knapsack_problem) and solves it using the following [`BranchAndBound`] algorithm.
//!
//! A solution is a tuple (or struct) `(J,i)` where `J` is a set of indices from `1..i` which represent all the already chosen items.
//! We bound a solution by computing its fractional optimal value using the last `i+1..n` items and the already chosen ones.
//!
//! Note that we keep an immutable reference to the problem instance in the solution itself to allow to dynamic access to the values.
//! Furthermore that this example is highly unoptimized.
//!
//! ```
//! use bnb::{BranchingOperator, BoundingOperator, BranchAndBound};
//!
//! #[derive(Clone, Debug)]
//! struct KnapsackInstance {
//!     num_items: usize,
//!     weights: Vec<f64>,
//!     values: Vec<f64>,
//!     weight_limit: f64,
//! }
//!
//! #[derive(Clone, Debug)]
//! struct KnapsackSolution<'a> {
//!     instance: &'a KnapsackInstance,
//!     chosen: Vec<bool>,
//!     index: usize,
//!     total_weight: f64,
//! }
//!
//! impl<'a> KnapsackSolution<'a> {
//!     pub fn new(ins: &'a KnapsackInstance) -> Self {
//!         Self {
//!             instance: ins,
//!             chosen: vec![],
//!             index: 0,
//!             total_weight: 0.0,
//!         }
//!     }
//! }
//!
//! impl<'a> BranchingOperator for KnapsackSolution<'a> {
//!     fn branch(&self) -> Vec<Self> {
//!         if self.index == self.instance.num_items {
//!             return vec![];
//!         }
//!
//!         let mut next_not_chosen: Vec<bool> = self.chosen.clone();
//!         next_not_chosen.push(false);
//!         let mut branches = vec![KnapsackSolution {
//!             instance: &self.instance,
//!             chosen: next_not_chosen,
//!             index: self.index + 1,
//!             total_weight: self.total_weight,
//!         }];
//!
//!         if self.total_weight + self.instance.weights[self.index] <= self.instance.weight_limit {
//!             let mut next_chosen: Vec<bool> = self.chosen.clone();
//!             next_chosen.push(true);
//!             branches.push(KnapsackSolution {
//!                 instance: &self.instance,
//!                 chosen: next_chosen,
//!                 index: self.index + 1,
//!                 total_weight: self.total_weight + self.instance.weights[self.index],
//!             });
//!         }
//!
//!         branches
//!     }
//! }
//!
//! impl<'a> BoundingOperator<f64> for KnapsackSolution<'a> {
//!     fn bound(&self) -> f64 {
//!         let mut bound: f64 = 0.0;
//!         for (i, b) in self.chosen.iter().enumerate() {
//!             if *b {
//!                 bound += self.instance.values[i];
//!             }
//!         }
//!
//!         let mut remaining_weight = self.instance.weight_limit - self.total_weight;
//!         let mut sorted_indices: Vec<(usize, f64)> = (self.index..self.instance.num_items)
//!             .into_iter()
//!             .map(|i| (i, self.instance.values[i] / self.instance.weights[i]))
//!             .collect();
//!         sorted_indices.sort_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap());
//!         loop {
//!             if let Some((i, _)) = sorted_indices.pop() {
//!                 if remaining_weight - self.instance.weights[i] < 0.0 {
//!                      let ratio = remaining_weight as f64 / self.instance.weights[i];
//!                     bound += ratio * self.instance.values[i];
//!                     break;
//!                 } else {
//!                     remaining_weight -= self.instance.weights[i];
//!                     bound += self.instance.values[i];
//!                 }
//!             } else {
//!                 break;
//!             }
//!         }
//!
//!         bound
//!     }
//!
//!     fn solution(&self) -> Option<f64> {
//!         if self.index < self.instance.num_items {
//!             return None;
//!         }
//!
//!         let mut total_value: f64 = 0.0;
//!         for (i, b) in self.chosen.iter().enumerate() {
//!             if *b {
//!                 total_value += self.instance.values[i];
//!             }
//!         }
//!         Some(total_value)
//!     }
//! }
//!
//!
//! let ins: KnapsackInstance = KnapsackInstance {
//!     num_items: 8,
//!     weights: vec![0.1, 0.4, 0.3, 0.7, 0.9, 0.2, 0.5, 0.6],
//!     values: vec![2.0, 3.1, 2.4, 0.9, 5.1, 0.8, 0.2, 4.0],
//!     weight_limit: 1.7,
//! };
//!
//! // Run Branch & Bound using DFS
//! let mut bnb = BranchAndBound::new(KnapsackSolution::new(&ins))
//!                 .maximize()
//!                 .use_dfs();
//!     
//! // Run the algorithm
//! bnb.run_to_completion();
//!
//! // The solution should exist
//! let sol = bnb.best_known_solution().unwrap();
//!
//! // Optimal value achieved us 12.3
//! assert_eq!(sol.0, 12.3);
//!
//! // Items 1,2,3,6,8 were chosen
//! assert_eq!(sol.1.chosen, vec![true, true, true, false, false, true, false, true]);
//! assert_eq!(sol.1.index, 8);
//!
//! // The total weight of chosen items is 1.6
//! assert_eq!(sol.1.total_weight, 1.6);
//!
//! // The algorithm took 21 iterations (77 for BestFirstSearch and 125 for BFS in this instance)
//! assert_eq!(bnb.num_iterations(), 21);
//!
//!
//!
//! ```

use seq::*;

pub mod seq;

// ------------------------------- //
//     Branch & Bound Operators    //
// ------------------------------- //

/// # Branch a set of solutions into (distinct) subsets of solutions
///
/// One of the two core building blocks of the [`BranchAndBound`] algorithm.
/// The [`BranchingOperator`] divides a set of solutions into (distinct) subsets of solutions.
/// These branches can then be evaluated and further branched or dismissed depending on the [`BoundingOperator`].
pub trait BranchingOperator: Sized {
    /// Based on `&self`, a number of branches (sub-solutions) are built.
    ///
    /// ## Implementation Notes
    ///
    /// - Keep a (immutable) reference to the problem instance and avoid cloning the instance if possible.
    /// - If the solution is a leaf (i.e. cannot be further branched from / is a set of solutions with cardinality 1), avoid panicking but instead return an empty vector.
    fn branch(&self) -> Vec<Self>;
}

/// # Bounds a set of solutions with a theoretical optimal value achievable by these solutions
///
/// One of the two core building blocks of the [`BranchAndBound`] algorithm.
/// The [`BoundingOperator`] evaluates a set of solutions and provides a theoretical lower/upper bound on the achievable values of these solutions.
/// Additionally, if a set of solutions contains exactly one solution (i.e. is a leaf node in the *solution-tree*), or is in any other way a *final*
/// solution, it can compute the achieved value of this specific solution.
pub trait BoundingOperator<V>
where
    V: PartialOrd,
{
    /// Computes a theoretical lower/upper bound on a set of solutions on the achievable values of these solutions.
    fn bound(&self) -> V;

    /// If a set of solutions is *final* (most often has cardinality 1), then this returns `Some(sol_value)` where
    /// `sol_value` is the achieved value of these solutions in this problem instance.
    ///
    /// ## Implementation Notes
    ///
    /// In many cases, this function might boil down to
    /// ```ignore
    /// fn solution(&self) -> Option<V> {
    ///     if ...self.is_final()... {
    ///         return Some(self.bound());
    ///     }
    ///     None
    /// }
    /// ```
    ///
    /// That might lead in some cases however to slightly higher runtimes when `self.bound()` is just an extension
    /// of `self.solution()`.
    /// See the example in the main documentation.
    fn solution(&self) -> Option<V>;
}

/// Placeholder struct to identify minimization problems.
pub struct BBMin();

/// Placeholder struct to identify maximization problems.
pub struct BBMax();

/// Trait defining whether a problem is a minimization problem or a maximization problem.
pub trait MinOrMax {
    /// Returns *true* if the problem is a maximization problem.
    fn is_maximize(&self) -> bool;
}

impl MinOrMax for BBMin {
    fn is_maximize(&self) -> bool {
        false
    }
}

impl MinOrMax for BBMax {
    fn is_maximize(&self) -> bool {
        true
    }
}

/// # Branch & Bound
///
/// The main algorithm and the heart of the crate.
pub struct BranchAndBound<V, B, S, M>
where
    V: PartialOrd + Clone,
    B: BranchingOperator + BoundingOperator<V>,
    S: NodeSequencer<BoundedSolution<V, B>>,
    M: MinOrMax,
{
    /// The starting node, i.e. the set of all solutions.
    start_node: B,

    /// The currently best obtained solution and value. This is must be a solution where `solution()` yields `Some(val)`.
    current_best_bound: Option<(V, B)>,

    /// The node sequencer holding all unprocessed nodes of solutions.
    node_sequencer: S,

    /// The goal type of the algorithm, i.e. minimization or maximization.
    goal_type: M,

    /// Number of iterations performed
    iterations: usize,
}

impl<V, B> BranchAndBound<V, B, Stack<BoundedSolution<V, B>>, BBMin>
where
    V: PartialOrd + Clone,
    B: BranchingOperator + BoundingOperator<V> + Clone,
{
    /// Creates a new [`BranchAndBound`] instance using a starting node, i.e. the set of all solutions.
    ///
    /// As default, every [`BranchAndBound`] algorithm is initialized as a minimization problem and using DFS.
    pub fn new(start_node: B) -> Self {
        Self {
            start_node: start_node.clone(),
            current_best_bound: None,
            node_sequencer: Stack::init(BoundedSolution::init(start_node)),
            goal_type: BBMin(),
            iterations: 0,
        }
    }
}

impl<V, B, S, M> BranchAndBound<V, B, S, M>
where
    V: PartialOrd + Clone,
    B: BranchingOperator + BoundingOperator<V>,
    S: NodeSequencer<BoundedSolution<V, B>>,
    M: MinOrMax,
{
    /// Consumes itself and returns a [`BranchAndBound`] algorithm with the same parameters but a minimization goal.
    ///
    /// Note that this function is obsolete since every [`BranchAndBound`] algorithm is initialized as a minimization problem.
    ///
    /// ## Warning
    /// Always call this method prior to `use_best_first_search()`!
    pub fn minimize(self) -> BranchAndBound<V, B, S, BBMin> {
        assert!(!self.node_sequencer.is_ordered());

        BranchAndBound {
            start_node: self.start_node,
            current_best_bound: self.current_best_bound,
            node_sequencer: self.node_sequencer,
            goal_type: BBMin(),
            iterations: self.iterations,
        }
    }

    /// Consumes itself and returns a [`BranchAndBound`] algorithm with the same parameters but a maximization goal.
    ///
    /// ## Warning
    /// Always call this method prior to `use_best_first_search()`!
    pub fn maximize(self) -> BranchAndBound<V, B, S, BBMax> {
        assert!(!self.node_sequencer.is_ordered());

        BranchAndBound {
            start_node: self.start_node,
            current_best_bound: self.current_best_bound,
            node_sequencer: self.node_sequencer,
            goal_type: BBMax(),
            iterations: self.iterations,
        }
    }

    /// Consumes itself and returns a [`BranchAndBound`] algorithm with the same parameters but it uses DFS to determine node-order.
    ///
    /// Note that this function is obsolete since every [`BranchAndBound`] algorithm is initialized with DFS as its node-sequencer.
    pub fn use_dfs(self) -> BranchAndBound<V, B, Stack<BoundedSolution<V, B>>, M> {
        BranchAndBound {
            start_node: self.start_node,
            current_best_bound: self.current_best_bound,
            node_sequencer: Stack::convert_from(self.node_sequencer),
            goal_type: self.goal_type,
            iterations: self.iterations,
        }
    }

    /// Consumes itself and returns a [`BranchAndBound`] algorithm with the same parameters but it uses BFS to determine node-order.
    pub fn use_bfs(self) -> BranchAndBound<V, B, Queue<BoundedSolution<V, B>>, M> {
        BranchAndBound {
            start_node: self.start_node,
            current_best_bound: self.current_best_bound,
            node_sequencer: Queue::convert_from(self.node_sequencer),
            goal_type: self.goal_type,
            iterations: self.iterations,
        }
    }

    /// Consumes itself and returns a [`BranchAndBound`] algorithm with the same parameters but with an additional start value and solution.
    ///
    /// It is common to compute such a solution and its value using a heuristic or other approximation algorithms.
    /// We expect a `value` field instead of calling `solution.solution()` to allow for something like `with_start_solution(B::default(), value)`
    /// if the heristic for example did not use the format of `B` and it is unimportant or too expensive to create it.
    pub fn with_start_solution(self, solution: B, value: V) -> Self {
        Self {
            start_node: self.start_node,
            current_best_bound: Some((value, solution)),
            node_sequencer: self.node_sequencer,
            goal_type: self.goal_type,
            iterations: self.iterations,
        }
    }
}

impl<V, B, S> BranchAndBound<V, B, S, BBMin>
where
    V: PartialOrd + Clone,
    B: BranchingOperator + BoundingOperator<V>,
    S: NodeSequencer<BoundedSolution<V, B>>,
{
    /// Consumes itself and returns a [`BranchAndBound`] algorithm with the same parameters but it uses Best-Fit-Search to determine node-order.
    pub fn use_best_first_search(
        self,
    ) -> BranchAndBound<V, B, PrioQueue<BoundedSolution<V, B>>, BBMin> {
        BranchAndBound {
            start_node: self.start_node,
            current_best_bound: self.current_best_bound,
            node_sequencer: PrioQueue::convert_from(self.node_sequencer),
            goal_type: self.goal_type,
            iterations: self.iterations,
        }
    }
}

impl<V, B, S> BranchAndBound<V, B, S, BBMax>
where
    V: PartialOrd + Clone,
    B: BranchingOperator + BoundingOperator<V>,
    S: NodeSequencer<BoundedSolution<V, B>>,
{
    /// Consumes itself and returns a [`BranchAndBound`] algorithm with the same parameters but it uses Best-Fit-Search to determine node-order.
    pub fn use_best_first_search(
        self,
    ) -> BranchAndBound<V, B, RevPrioQueue<BoundedSolution<V, B>>, BBMax> {
        BranchAndBound {
            start_node: self.start_node,
            current_best_bound: self.current_best_bound,
            node_sequencer: RevPrioQueue::convert_from(self.node_sequencer),
            goal_type: self.goal_type,
            iterations: self.iterations,
        }
    }
}

impl<V, B, S, M> BranchAndBound<V, B, S, M>
where
    V: PartialOrd + Clone,
    B: BranchingOperator + BoundingOperator<V>,
    S: NodeSequencer<BoundedSolution<V, B>>,
    M: MinOrMax,
{
    /// Executes the next step in the computation.
    pub fn execute_step(&mut self) {
        self.iterations += 1;
        // If there is a (next) node in the sequencer that has a better theoretical bound than the currently best known solution (if existing), then...
        if let Some(node) = self.node_sequencer.pop_until_satisfied(|node| {
            if let Some((val, _)) = self.current_best_bound.as_ref() {
                if (self.goal_type.is_maximize() && node.value() <= *val)
                    || (!self.goal_type.is_maximize() && node.value() >= *val)
                {
                    return false;
                }
            }
            true
        }) {
            // ...branch the node and...
            let (_, node) = node.split();
            for branch_node in node.branch() {
                // ... if the node is *final* and represents exactly one solution. If then, check if we have to update the currently best known or solution or not and continue...
                if let Some(sol) = branch_node.solution() {
                    if let Some((val, _)) = self.current_best_bound.as_ref() {
                        if (self.goal_type.is_maximize() && sol > *val)
                            || (!self.goal_type.is_maximize() && sol < *val)
                        {
                            self.current_best_bound = Some((sol, branch_node));
                        }
                    } else {
                        self.current_best_bound = Some((sol, branch_node));
                    }
                    continue;
                }
                // ... If not, bound the node and into the sequencer if its bound is not worse than the currently best known solution.
                let bound = branch_node.bound();
                if let Some((val, _)) = self.current_best_bound.as_ref() {
                    if (self.goal_type.is_maximize() && bound <= *val)
                        || (!self.goal_type.is_maximize() && bound >= *val)
                    {
                        continue;
                    }
                }
                self.node_sequencer
                    .push(BoundedSolution::new(bound, branch_node));
            }
        }
    }

    /// Returns *true* if the execution is finished and the best solution was found.
    pub fn is_completed(&self) -> bool {
        self.node_sequencer.len() == 0
    }

    /// Runs the algorithm until it is completed, i.e. when `self.is_completed()` yields *true*.
    pub fn run_to_completion(&mut self) {
        while !self.is_completed() {
            self.execute_step();
        }
    }

    /// Returns the currently best known solution at any point in the computation (might be `None` if there was no found yet).
    pub fn best_known_solution(&self) -> Option<&(V, B)> {
        if let Some(sol) = self.current_best_bound.as_ref() {
            return Some(sol);
        }
        None
    }

    /// Returns the current number of iterations performed by the algorithm.
    pub fn num_iterations(&self) -> usize {
        self.iterations
    }
}
