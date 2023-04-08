# Branch & Bound 

This crate provides a general template for [Branch & Bound](https://en.wikipedia.org/wiki/Branch_and_bound) algorithms.

The base building block is the `BranchingOperator` trait and the `BoundingOperator` trait. Implement these for your custom structs that represent solutions and use the `BranchAndBound` struct to run the algorithm.

## Usage

Include the following into your `Cargo.toml` file:

```toml
[dependencies]
bnb = "0.1.1"
```

To implement your own `BranchAndBound` algorithm, import the `BranchingOperator` and the `BoundingOperator` trait and implement them for your own (solution) struct.
Note that for the `BoundingOperator<V>` trait, you might use any type or struct that implements the `PartialOrd` trait as `V`.

```rust
use bnb::{BranchingOperator, BoundingOperator};

struct YourStruct { ... }

impl BranchingOperator for YourStruct {
    fn branch(&self) -> Vec<Self> {
        ...
    }
}

impl BoundingOperator<u32> for YourStruct {
    fn bound(&self) -> u32 {
        ...
    }

    fn solution(&self) -> Option<u32> {
        ...
    }
}
```

Afterwards, import the `BranchAndBound` struct and the `IterativeAlgorithm` trait and initialize the `BranchAndBound` struct using a starting state version of `YourStruct`:

```rust
use bnb::{BranchingOperator, BoundingOperator, BranchAndBound, IterativeAlgorithm};

struct YourStruct { ... }

...

fn main() {
    let mut bnb = BranchAndBound::new(...) 
                    .minimize() // The problem is minimization problem
                    .use_dfs() // Use DFS to find the next node in the BnB-Tree
                    .with_start_solution(...); // Optional initialization with a starting solution 

    // Run the algorithm
    bnb.run_to_completion();

    // The solution should exist
    let sol = bnb.best_known_solution().unwrap();

    ...
}

```

Alternatively, you can just abbreviate to 
```rust
use bnb::*;
```
to import all neccessary features. This only imports the additional `seq`-submodule which is not needed for external use.

See the documentation for a more extensive example of the [Knapsack-Problem](https://en.wikipedia.org/wiki/Knapsack_problem).
