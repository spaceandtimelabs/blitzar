# Benchmarks for the multiproduct computation

## Description

In this benchmark, we compare both the naive and the pippenger multiproduct implementation.

## Running the benchmark

```
benchmark/multiprod1/benchmark.sh <use_naive>

# `use_naive` must be 0 or 1. When set to 1, the naive multiproduct is used. When set to 0, the Pippenger multiproduct is used.
```

Example:

```
benchmark/multiprod1/benchmark.sh 0
```

During this execution, some files are generated in the `benchmark/multiprod1/.results/` directory.

For more information regarding the Pippenger multiproduct, check [here](https://cacr.uwaterloo.ca/techreports/2010/cacr2010-26.pdf).
