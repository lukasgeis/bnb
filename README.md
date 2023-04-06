# Branch & Bound 

This crate provides a general template for [Branch & Bound](https://en.wikipedia.org/wiki/Branch_and_bound) algorithms.

The base building block is the `BranchingOperator` trait and the `BoundingOperator` trait. Implement these for your custom structs that represent solutions and use the `BranchAndBound` struct to run the algorithm.

See the documentation for more details and an example.