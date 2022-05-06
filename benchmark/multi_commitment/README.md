**Benchmarking for cpu vs gpu of naive commitment computation**

## Description

Given the number of rows, the number of columns, and the number of bytes used to represent numbers in a table with dimension rows x columns, we compute the SNARK commitments of each column, using for that curve25519 and ristretto group theory. Though we specify the number of rows, columns, and bytes per cell, the values of the table are randomly generated (in a deterministic way, given that the same seed is always provided).

## Setup

Temporarily, this setup uses an Intel(R) Core(TM) i7-7700HQ CPU @ 2.80GHz 2.81 GHz. For each run, we measured the duration in seconds and the throughput (word_size * cols * rows / <cpu_duration | gpu_duation>). But we intend to use in the future an Azure VM [Standard_NC6s_v3](https://docs.microsoft.com/en-us/azure/virtual-machines/ncv3-series) with a V100 GPU.

In each run of the below table, we changed the number of rows and columns of the input table used in the commitment computation, thus varying the workload of the CPU / GPU execution. Currently, we only have the CPU implementation; thus the incomplete values for the GPU fields.

To run the entire benchmark, execute:

> ./benchmark/multi_commitment/run_benchmark.sh

## Results

| Cols x Rows | Word Size | Table Size (MB) | CPU Duration (s) | CPU Throughput (bytes / s) | GPU Duration (s) | GPU Throughput (bytes / s) | Duration Speedup (GPU / CPU) |
| ----------- | --------- | --------------- | ---------------- | -------------------------- | ---------------- | -------------------------- | ---------------------------- |
| 10x10       | 4         | 0.000           | 0.008            | 5.08E+04                   |                  |                            |                              |
| 10x100      | 4         | 0.004           | 0.072            | 5.52E+04                   |                  |                            |                              |
| 10x1000     | 4         | 0.038           | 0.764            | 5.24E+04                   |                  |                            |                              |
| 100x10      | 4         | 0.004           | 0.074            | 5.38E+04                   |                  |                            |                              |
| 100x100     | 4         | 0.038           | 0.756            | 5.29E+04                   |                  |                            |                              |
| 100x1000    | 4         | 0.381           | 7.114            | 5.62E+04                   |                  |                            |                              |
| 1000x10     | 4         | 0.038           | 0.745            | 5.37E+04                   |                  |                            |                              |
| 1000x100    | 4         | 0.381           | 7.733            | 5.17E+04                   |                  |                            |                              |
| 1000x1000   | 4         | 3.815           | 76.33            | 5.24E+04                   |                  |                            |                              |