# Benchmarks for the commitment computation

## Description

Given the number of each commitment length, the total number of commitments, and the number of bytes used to represent data elements, we compute each commitment sequence, using for that curve25519 and ristretto group operations. The data elements are randomly generated (in a deterministic way, given that the same seed is always provided).

## Running the benchmarks

To run the whole benchmark on the GPU (update accordingly to use CPU), execute:

```
docker run --rm -e TEST_TMPDIR=/root/.cache_bazel -v /home/joe/Documents/proofs-gpu/benchmark/multi_commitment:/root/.cache_bazel -v "$PWD":/src -w /src --gpus all --privileged -it joestifler/proofs_gpu:7.0 benchmark/multi_commitment/scripts/run_benchmark.py --backend gpu --output-dir benchmark/multi_commitment/.proof_results --force-rerun-bench 1 --run-bench-callgrind 1 --run-bench 1
```

Some files are generated in this process. They can be found on `benchmark/multi_commitment/.proof_results/` directory.
