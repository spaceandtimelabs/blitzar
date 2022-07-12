**Benchmarks for the commitment computation**

## Description

Given the number of each commitment length, the total number of commitments, and the number of bytes used to represent data elements, we compute each commitment sequence, using for that curve25519 and ristretto group points. The data elements are randomly generated (in a deterministic way, given that the same seed is always provided).

# Benchmarks

We use five different data types for our data elements:

- 0 byte: they represent random numbers from the range 0 to 1 containing 1 byte. This is the boolean case.
- 1 byte: they represent random numbers from the range 0 to 255 containing 1 byte.
- 4 byte: they represent random numbers from the range 0 to (2**32 - 1) containing 4 bytes.
- 8 byte: they represent random numbers from the range 0 to (2**64 - 1) containing 8 bytes.
- 32 byte: they represent random numbers from the range 0 to (2**256 - 1) containing 32 bytes.

We also benchmark three different backends that we have available:

1. `naive-cpu`: this is the naive cpu implementation of the commitment computation
2. `naive-gpu`: this is the naive gpu implementation of the commitment computation
2. `pip-gpu`: this is the improved cpu implementation of the commitment computation, using for that the [Pippenger algorithm](https://cacr.uwaterloo.ca/techreports/2010/cacr2010-26.pdf).

Each benchmark result that will be presented was run at least 3x and the average time value was taken.

## Setup

This experiment uses an Azure VM [Standard_NC6s_v3](https://docs.microsoft.com/en-us/azure/virtual-machines/ncv3-series) Intel(R) Xeon(R) CPU E5-2690 v4 @ 2.60GHz (6 cores), 110GB Ram, powered with one Tesla V100 16 GB of memory. For each run, we measured the duration in seconds and the throughput (word_size * num_commitments * commitment_length / <cpu_duration | gpu_duration>).

In each run of the below table, we changed the number of commitments as well as the commitments length of the input table, thus varying the workload of the CPU / GPU execution.

To run the entire benchmark, execute:

> ./benchmark/multi_commitment/benchmark.sh <pip-cpu | naive-cpu | naive-gpu> <run>

For the current code version, you can check the benchmarks in the directory `benchmark_results` under:

1. `benchmark_results/summary_naive-cpu.txt`
2. `benchmark_results/summary_naive-gcpu.txt`
3. `benchmark_results/summary_pip-cpu.txt`

You can copy and past the above file contents inside a google spreadsheet. With that, it may become clear to analyze their results.

To generate benchmark plots, just run: `python3 plot_throughput`. An SVG file will be generated under the `benchmark_results` directory.

