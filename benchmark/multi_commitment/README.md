# Benchmarks for the commitment computation

## Description

Given the number of each commitment length, the total number of commitments, and the number of bytes used to represent data elements, we compute each commitment sequence, using for that curve25519 and ristretto group operations. The data elements are randomly generated (in a deterministic way, given that the same seed is always provided).

# Benchmarks

We use five different data types for our data elements:

- 0 byte: they represent random 1-byte numbers from the range 0 to 1. This is the boolean case.
- 1 byte: they represent random 1-byte numbers from the range 0 to 255.
- 4 byte: they represent random 4-byte numbers from the range 0 to (2**32 - 1).
- 8 byte: they represent random 8-byte numbers from the range 0 to (2**64 - 1).
- 32 byte: they represent random 32-byte numbers from the range 0 to (2**256 - 1).

We also benchmark three different backends that we have available:

1. `naive-cpu`: this is the naive cpu commitment computation
2. `naive-gpu`: this is the naive gpu commitment computation
2. `pip-gpu`: this is the improved cpu commitment computation, using partially for that the [Pippenger algorithm](https://cacr.uwaterloo.ca/techreports/2010/cacr2010-26.pdf).

## Setup

This experiment uses an Azure VM [Standard_NC6s_v3](https://docs.microsoft.com/en-us/azure/virtual-machines/ncv3-series) Intel(R) Xeon(R) CPU E5-2690 v4 @ 2.60GHz (6 cores), 110GB Ram, powered with one Tesla V100 16 GB of memory. For each run, we measured the duration in seconds and the throughput (word_size * num_commitments * commitment_length / <cpu_duration | gpu_duration>). Each benchmark run at least 3x and the average time value is taken. In each run, we changed the number of commitments as well as the commitments length of the input table, thus varying the workload of the CPU / GPU execution. All the results can be found in the `benchmark_results` directory.

## Rerun the benchmarks

To run the whole benchmark again, execute:

```
./benchmark/multi_commitment/benchmark.sh <pip-cpu | naive-cpu | naive-gpu> <run-benchmark-suite> <run-callgrind-suite>

# `run-benchmark-suite` must be 0 or 1. When set to 1, the whole benchmark suite is run, and a summary text file is generated.
# `run-callgrind-suite` must be 0 or 1. When set to 1, some callgrind files are generated, which are related to a reduced benchmark suite.
```

Example:

```
./benchmark/multi_commitment/benchmark.sh pip-cpu 1 1
```

Some files are generated in this process. They can be found on `benchmark/multi_commitment/.results/ directory`. When the `run-benchmark-suite` is set to 1 and `naive-cpu` backend is used, the file `benchmark/multi_commitment/.results/summary_naive-cpu.txt` with all the summarized information will be generated. Similarly, when the backend is different, the summary file name will be changed accordingly. To improve visualization, you can copy and paste its content to a Google spreadsheet table. To plot speedup graphs based on the previous summary files, use the following script:

```
./benchmark/multi_commitment/plot_throughput.py
```

An SVG file is generated under the `benchmark/multi_commitment/.results/` directory as a by-product of this python execution. Bear in mind that you need to have previously executed the following commands:

```
./benchmark/multi_commitment/benchmark.sh pip-cpu 1 0
./benchmark/multi_commitment/benchmark.sh naive-cpu 1 0
```

as they are responsible for generating the summary text files used by the python script.
