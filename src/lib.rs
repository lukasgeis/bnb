//! # Branch & Bound Template
//!
//! This crate provides a general template for Branch & Bound algorithms.
//!

use std::{
    cmp::{Ordering, Reverse},
    collections::{BinaryHeap, VecDeque},
};

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
pub trait BoundingOperator<V> {
    /// Computes a theoretical lower/upper bound on a set of solutions on the achievable values of these solutions.
    fn bound(&self) -> V;

    /// If a set of solutions is *final* (most often has cardinality 1), then this returns `Some(sol_value)` where
    /// `sol_value` is the achieved value of these solutions in this problem instance.
    ///
    /// ## Implementation Notes
    ///
    /// In many cases, this function might boil down to
    /// ```no_run
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

// ------------------------------- //
//        Node - Sequencer         //
// ------------------------------- //

/// # Push & Pop Nodes
///
/// This trait provides basic methods describing the functionality of a data structure
/// storing values (nodes) of any type T in a specific sequential manner.
/// Many of these methods are already implemented for most structures like `Vec<T>`, but
/// are refurbished in order to use them interchangibly in the [`BranchAndBound`] algorithm.
pub trait NodeSequencer<T>: Sized {
    /// Initializes the sequencer with a starting object as the first element.
    fn init(item: T) -> Self;

    /// Pushes a given element onto the sequencer.
    fn push(&mut self, item: T);

    /// Pops the highest-order-priority element from the sequencer and returns it as an `Option<T>`.
    /// Returns `None` if the sequencer was empty.
    fn pop(&mut self) -> Option<T>;

    /// Returns the number of elements in the sequencer.
    fn len(&self) -> usize;

    /// Consumes the sequencer and returns a `Vec<T>` with all remaining objects in the sequencer.
    fn to_vec(self) -> Vec<T>;

    /// Creates a sequencer from a `Vec<T>`.
    /// Note that elements will most likely be pushed into the sequencer in the given order in the vector.
    fn from_vec(items: Vec<T>) -> Self;

    /// Consumes a foreign sequencer to create a new sequencer of this type.
    ///
    /// Note that this is only meant to prevent writing `Self::from_vec(other.to_vec())` everytime.
    fn convert_from<S>(other: S) -> Self
    where
        S: NodeSequencer<T>,
    {
        Self::from_vec(other.to_vec())
    }

    /// Keeps popping elements from the sequencer until a condition is met or the sequencer is empty.
    fn pop_until_satisfied<F>(&mut self, f: F) -> Option<T>
    where
        F: Fn(&T) -> bool,
    {
        while let Some(item) = self.pop() {
            if f(&item) {
                return Some(item);
            }
        }
        None
    }
}

/// # LIFO
///
/// This sequencer is used for the DFS (Depth-First-Search) algorithm.
pub type Stack<T> = Vec<T>;

/// # FIFO
///
/// This sequencer is used for the BFS (Breadth-First-Search) algorithm.
pub type Queue<T> = VecDeque<T>;

/// # Min-Heap
///
/// This sequencer is used for the Best-First-Algorithms in minimization problems.
pub type PrioQueue<T> = BinaryHeap<T>;

/// # Max-Heap
///
/// This sequencer is used for the Best-First-Algorithms in maximization problems.
pub struct RevPrioQueue<T>(BinaryHeap<Reverse<T>>);

impl<T> NodeSequencer<T> for Stack<T> {
    fn init(item: T) -> Self {
        vec![item]
    }

    fn push(&mut self, item: T) {
        self.push(item)
    }

    fn pop(&mut self) -> Option<T> {
        self.pop()
    }

    fn len(&self) -> usize {
        self.len()
    }

    fn to_vec(self) -> Vec<T> {
        self
    }

    fn from_vec(items: Vec<T>) -> Self {
        items
    }
}

impl<T> NodeSequencer<T> for Queue<T> {
    fn init(item: T) -> Self {
        Self::from(vec![item])
    }

    fn push(&mut self, item: T) {
        self.push_back(item)
    }

    fn pop(&mut self) -> Option<T> {
        self.pop_front()
    }

    fn len(&self) -> usize {
        self.len()
    }

    fn to_vec(self) -> Vec<T> {
        self.into()
    }

    fn from_vec(items: Vec<T>) -> Self {
        Self::from(items)
    }
}

impl<T> NodeSequencer<T> for PrioQueue<T>
where
    T: Ord,
{
    fn init(item: T) -> Self {
        Self::from(vec![item])
    }

    fn push(&mut self, item: T) {
        self.push(item)
    }

    fn pop(&mut self) -> Option<T> {
        self.pop()
    }

    fn len(&self) -> usize {
        self.len()
    }

    fn to_vec(self) -> Vec<T> {
        self.into_vec()
    }

    fn from_vec(items: Vec<T>) -> Self {
        Self::from(items)
    }
}

impl<T> NodeSequencer<T> for RevPrioQueue<T>
where
    T: Ord,
{
    fn init(item: T) -> Self {
        RevPrioQueue(BinaryHeap::from(vec![Reverse(item)]))
    }

    fn push(&mut self, item: T) {
        self.0.push(Reverse(item))
    }

    fn pop(&mut self) -> Option<T> {
        if let Some(Reverse(item)) = self.0.pop() {
            return Some(item);
        }
        None
    }

    fn len(&self) -> usize {
        self.0.len()
    }

    fn to_vec(self) -> Vec<T> {
        self.0.into_iter().map(|Reverse(item)| item).collect()
    }

    fn from_vec(items: Vec<T>) -> Self {
        RevPrioQueue(BinaryHeap::from(
            items
                .into_iter()
                .map(|item| Reverse(item))
                .collect::<Vec<Reverse<T>>>(),
        ))
    }
}

/// Struct representing a solution (or node) and its bound.
/// This is mainly used to avoid repeatedly calling `bound()` on a solution and it allows
/// for easier handling in [`PrioQueue`] and [`RevPrioQueue`].
#[derive(PartialEq)]
pub struct BoundedNode<V, B>
where
    V: PartialOrd + Clone,
    B: BranchingOperator + BoundingOperator<V> + PartialEq,
{
    /// Bound of the solution.
    bound: V,

    /// The solution itself.
    node: B,
}

/// When `Ord` is implemented for `V` (and `Eq` is implemented for `B`), then we can also implement `Eq` for `BoundedNode<V, B>`.
impl<V, B> Eq for BoundedNode<V, B>
where
    V: Ord + Clone,
    B: BranchingOperator + BoundingOperator<V> + Eq,
{
}

/// When `Ord` is implemented for `V` (and `Eq` is implemented for `B`), then we can also implement `Ord` for `BoundedNode<V, B>`.
impl<V, B> Ord for BoundedNode<V, B>
where
    V: Ord + Clone,
    B: BranchingOperator + BoundingOperator<V> + Eq,
{
    fn cmp(&self, other: &Self) -> Ordering {
        self.bound.cmp(&other.bound)
    }
}

impl<V, B> PartialOrd for BoundedNode<V, B>
where
    V: PartialOrd + Clone,
    B: BranchingOperator + BoundingOperator<V> + PartialEq,
{
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.bound.partial_cmp(&other.bound)
    }
}

impl<V, B> BoundedNode<V, B>
where
    V: PartialOrd + Clone,
    B: BranchingOperator + BoundingOperator<V> + PartialEq,
{
    /// Creates a new [`BoundedNode`] using a given bound and a given solution.
    pub fn new(bound: V, node: B) -> Self {
        Self {
            bound: bound,
            node: node,
        }
    }

    /// Initializes a [`BoundedNode`] from a solution using the `bound()` method.
    pub fn init(node: B) -> Self {
        Self {
            bound: node.bound(),
            node: node,
        }
    }

    /// Consumes the [`BoundedNode`] and returns the bound and the solution a s a tuple.
    pub fn split(self) -> (V, B) {
        (self.bound, self.node)
    }

    /// Clones the bound and returns it.
    pub fn value(&self) -> V {
        self.bound.clone()
    }
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
    B: BranchingOperator + BoundingOperator<V> + PartialEq,
    S: NodeSequencer<BoundedNode<V, B>>,
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
}

impl<V, B> BranchAndBound<V, B, Stack<BoundedNode<V, B>>, BBMin>
where
    V: PartialOrd + Clone,
    B: BranchingOperator + BoundingOperator<V> + PartialEq + Clone,
{
    /// Creates a new [`BranchAndBound`] instance using a starting node, i.e. the set of all solutions.
    ///
    /// As default, every [`BranchAndBound`] algorithm is initialized as a minimization problem and using DFS.
    pub fn new(start_node: B) -> Self {
        Self {
            start_node: start_node.clone(),
            current_best_bound: None,
            node_sequencer: Stack::init(BoundedNode::init(start_node)),
            goal_type: BBMin(),
        }
    }
}

impl<V, B, S, M> BranchAndBound<V, B, S, M>
where
    V: PartialOrd + Clone,
    B: BranchingOperator + BoundingOperator<V> + PartialEq,
    S: NodeSequencer<BoundedNode<V, B>>,
    M: MinOrMax,
{
    /// Consumes itself and returns a [`BranchAndBound`] algorithm with the same parameters but a minimization goal.
    ///
    /// Note that this function is obsolete since every [`BranchAndBound`] algorithm is initialized as a minimization problem.
    pub fn minimize(self) -> BranchAndBound<V, B, S, BBMin> {
        BranchAndBound {
            start_node: self.start_node,
            current_best_bound: self.current_best_bound,
            node_sequencer: self.node_sequencer,
            goal_type: BBMin(),
        }
    }

    /// Consumes itself and returns a [`BranchAndBound`] algorithm with the same parameters but a maximization goal.
    pub fn maximize(self) -> BranchAndBound<V, B, S, BBMax> {
        BranchAndBound {
            start_node: self.start_node,
            current_best_bound: self.current_best_bound,
            node_sequencer: self.node_sequencer,
            goal_type: BBMax(),
        }
    }

    /// Consumes itself and returns a [`BranchAndBound`] algorithm with the same parameters but it uses DFS to determine node-order.
    ///
    /// Note that this function is obsolete since every [`BranchAndBound`] algorithm is initialized with DFS as its node-sequencer.
    pub fn use_dfs(self) -> BranchAndBound<V, B, Stack<BoundedNode<V, B>>, M> {
        BranchAndBound {
            start_node: self.start_node,
            current_best_bound: self.current_best_bound,
            node_sequencer: Stack::convert_from(self.node_sequencer),
            goal_type: self.goal_type,
        }
    }

    /// Consumes itself and returns a [`BranchAndBound`] algorithm with the same parameters but it uses BFS to determine node-order.
    pub fn use_bfs(self) -> BranchAndBound<V, B, Queue<BoundedNode<V, B>>, M> {
        BranchAndBound {
            start_node: self.start_node,
            current_best_bound: self.current_best_bound,
            node_sequencer: Queue::convert_from(self.node_sequencer),
            goal_type: self.goal_type,
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
        }
    }
}

impl<V, B, S> BranchAndBound<V, B, S, BBMin>
where
    V: Ord + Clone,
    B: BranchingOperator + BoundingOperator<V> + Eq,
    S: NodeSequencer<BoundedNode<V, B>>,
{
    /// Consumes itself and returns a [`BranchAndBound`] algorithm with the same parameters but it uses Best-Fit-Search to determine node-order.
    pub fn use_best_first_search(
        self,
    ) -> BranchAndBound<V, B, PrioQueue<BoundedNode<V, B>>, BBMin> {
        BranchAndBound {
            start_node: self.start_node,
            current_best_bound: self.current_best_bound,
            node_sequencer: PrioQueue::convert_from(self.node_sequencer),
            goal_type: self.goal_type,
        }
    }
}

impl<V, B, S> BranchAndBound<V, B, S, BBMax>
where
    V: Ord + Clone,
    B: BranchingOperator + BoundingOperator<V> + Eq,
    S: NodeSequencer<BoundedNode<V, B>>,
{
    /// Consumes itself and returns a [`BranchAndBound`] algorithm with the same parameters but it uses Best-Fit-Search to determine node-order.
    pub fn use_best_first_search(
        self,
    ) -> BranchAndBound<V, B, RevPrioQueue<BoundedNode<V, B>>, BBMax> {
        BranchAndBound {
            start_node: self.start_node,
            current_best_bound: self.current_best_bound,
            node_sequencer: RevPrioQueue::convert_from(self.node_sequencer),
            goal_type: self.goal_type,
        }
    }
}

/// Trait defining an iterative algorithm to allow step for step execution of the [`BranchAndBound`] algorithm.
pub trait IterativeAlgorithm {
    /// The type of solution, the algorithm will return.
    type Solution;

    /// Executes the next step in the computation.
    ///
    /// Might panic if `self.is_completed()` yields *true*.
    fn execute_step(&mut self);

    /// Returns *true* if the execution is finished and the best solution was found.
    fn is_completed(&self) -> bool;

    /// Returns the currently best known solution at any point in the computation (might be `None` if there was no found yet).
    fn best_known_solution(&self) -> Option<&Self::Solution>;

    /// Runs the algorithm until it is completed, i.e. when `self.is_completed()` yields *true*.
    /// Returns the best solution found in the process or `None` if none was found.
    fn run_to_completion(&mut self) -> Option<&Self::Solution> {
        while !self.is_completed() {
            self.execute_step();
        }
        self.best_known_solution()
    }
}

impl<V, B, S, M> IterativeAlgorithm for BranchAndBound<V, B, S, M>
where
    V: PartialOrd + Clone,
    B: BranchingOperator + BoundingOperator<V> + PartialEq,
    S: NodeSequencer<BoundedNode<V, B>>,
    M: MinOrMax,
{
    type Solution = (V, B);

    fn execute_step(&mut self) {
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
                    .push(BoundedNode::new(bound, branch_node));
            }
        }
    }

    fn is_completed(&self) -> bool {
        self.node_sequencer.len() == 0
    }

    fn best_known_solution(&self) -> Option<&Self::Solution> {
        if let Some(sol) = self.current_best_bound.as_ref() {
            return Some(sol);
        }
        None
    }
}
