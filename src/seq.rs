//! # Node Sequencing
//!
//! This submodule provides a template for types of node-sequencers and their implementations for the following:
//! - Stack (Last-In-First-Out)
//! - Queue (First-In-First-Out)
//! - PrioQueue & RevPrioQueue (Best-Out depending on whether best is minimal or maximal)

use std::{
    cmp::{Ordering, Reverse},
    collections::{BinaryHeap, VecDeque},
};

use crate::{BoundingOperator, BranchingOperator};

/// # Push & Pop Nodes
///
/// This trait provides basic methods describing the functionality of a data structure
/// storing values (nodes) of any type T in a specific sequential manner.
/// Many of these methods are already implemented for most structures like `Vec<T>`, but
/// are refurbished in order to use them interchangibly in the [BranchAndBound](super::BranchAndBound) algorithm.
pub trait NodeSequencer<T>: Sized {
    /// Returns *true* if the sequence is ordered and minimum/maximum matter.
    fn is_ordered(&self) -> bool;

    /// Initializes the sequencer with a starting object as the first element.
    fn init(item: T) -> Self;

    /// Pushes a given element onto the sequencer.
    fn push(&mut self, item: T);

    /// Pops the highest-order-priority element from the sequencer and returns it as an `Option<T>`.
    /// Returns `None` if the sequencer was empty.
    fn pop(&mut self) -> Option<T>;

    /// Returns the number of elements in the sequencer.
    fn len(&self) -> usize;

    /// Returns *true* if the length is zero
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

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
    fn is_ordered(&self) -> bool {
        false
    }

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
    fn is_ordered(&self) -> bool {
        false
    }

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
    fn is_ordered(&self) -> bool {
        true
    }

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
    fn is_ordered(&self) -> bool {
        true
    }

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

/// Struct representing a solution and its bound.
/// This is mainly used to avoid repeatedly calling `bound()` on a solution and it allows
/// for easier handling in [`PrioQueue`] and [`RevPrioQueue`].
pub struct BoundedSolution<V, B>
where
    V: PartialOrd + Clone,
    B: BranchingOperator + BoundingOperator<V>,
{
    /// Bound of the solution.
    bound: V,

    /// The solution itself.
    solution: B,
}

impl<V, B> PartialEq for BoundedSolution<V, B>
where
    V: PartialOrd + Clone,
    B: BranchingOperator + BoundingOperator<V>,
{
    fn eq(&self, other: &Self) -> bool {
        self.partial_cmp(other) == Some(Ordering::Equal)
    }
}

impl<V, B> Eq for BoundedSolution<V, B>
where
    V: PartialOrd + Clone,
    B: BranchingOperator + BoundingOperator<V>,
{
}

impl<V, B> PartialOrd for BoundedSolution<V, B>
where
    V: PartialOrd + Clone,
    B: BranchingOperator + BoundingOperator<V>,
{
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<V, B> Ord for BoundedSolution<V, B>
where
    V: PartialOrd + Clone,
    B: BranchingOperator + BoundingOperator<V>,
{
    fn cmp(&self, other: &Self) -> Ordering {
        match self.bound.partial_cmp(&other.bound) {
            Some(ordering) => ordering,
            None => Ordering::Equal,
        }
    }
}

impl<V, B> BoundedSolution<V, B>
where
    V: PartialOrd + Clone,
    B: BranchingOperator + BoundingOperator<V>,
{
    /// Creates a new [`BoundedSolution`] using a given bound and a given solution.
    pub fn new(bound: V, node: B) -> Self {
        Self {
            bound,
            solution: node,
        }
    }

    /// Initializes a [`BoundedSolution`] from a solution using the `bound()` method.
    pub fn init(node: B) -> Self {
        Self {
            bound: node.bound(),
            solution: node,
        }
    }

    /// Consumes the [`BoundedSolution`] and returns the bound and the solution a s a tuple.
    pub fn split(self) -> (V, B) {
        (self.bound, self.solution)
    }

    /// Clones the bound and returns it.
    pub fn value(&self) -> V {
        self.bound.clone()
    }
}
