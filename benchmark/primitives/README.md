# Benchmarking for primitives on the GPU
Currently only supports benchmarks for addition with `bls12-381` `G1` curve elements. Creates two vectors of size `<vector_size>` and does vector addition with them `<repetitions>` number of times. Timing results are output in microseconds.

This benchmark is similar to the Icicle's [example/multiply](https://github.com/ingonyama-zk/icicle/tree/40309329fbf6c5fc7e77d629c72b4a3d28036444/examples/c%2B%2B/multiply), but updated to measure curve addition operations.

## Usages
`bazel run -c opt //benchmark/primitives:benchmark <vector_size> <repetitions>`

## Example
`bazel run -c opt //benchmark/primitives:benchmark bls12_381 field 1000000 1000 256 10`
`bazel run -c opt //benchmark/primitives:benchmark bls12_381 curve 10000 1000 256 10`