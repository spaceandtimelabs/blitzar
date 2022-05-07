**Benchmarking for cpu vs gpu of naive commitment computation**

## Description

Given the number of rows, the number of columns, and the number of bytes used to represent numbers in a table with dimension rows x columns, we compute the SNARK commitments of each column, using for that curve25519 and ristretto group theory. Though we specify the number of rows, columns, and bytes per cell, the values of the table are randomly generated (in a deterministic way, given that the same seed is always provided).

## Setup

This experiment uses an Azure VM [Standard_NC6s_v3](https://docs.microsoft.com/en-us/azure/virtual-machines/ncv3-series) Intel(R) Xeon(R) CPU E5-2690 v4 @ 2.60GHz (6 cores), 110GB Ram, powered with one Tesla V100 16 GB of memory. For each run, we measured the duration in seconds and the throughput (word_size * cols * rows / <cpu_duration | gpu_duration>).

In each run of the below table, we changed the number of rows and columns of the input table used in the commitment computation, thus varying the workload of the CPU / GPU execution.

To run the entire benchmark, execute:

> ./benchmark/multi_commitment/run_benchmark.sh <cpu | gpu>

To compare cpu with gpu results, execute:

> ./benchmark/multi_commitment/diff_cpu_gpu.sh

Headers may appear in this diff, but no commitment result array should appear. If some commitment array appears, then gpu and cpu results are mismatching.

## Results

| Cols x Rows | Word Size | Table Size (MB) | CPU Duration (s) | CPU Throughput (bytes / s) | GPU Duration (s) | GPU Throughput (bytes / s) | Duration Speedup (GPU / CPU) |
| ----------- | --------- | --------------- | ---------------- | -------------------------- | ---------------- | -------------------------- | ---------------------------- |
| 10x10       | 32        | 0.003           | 0.009            | 3.52E-01                   | 0.029228         | 1.09E+05                   | 0.30x                        |
| 10x100      | 32        | 0.031           | 0.082            | 3.71E-01                   | 0.269627         | 1.19E+05                   | 0.30x                        |
| 10x1000     | 32        | 0.305           | 0.817            | 3.73E-01                   | 2.577824         | 1.24E+05                   | 0.32x                        |
| ----------- | --------- | --------------- | ---------------- | -------------------------- | ---------------- | -------------------------- | ---------------------------- |
| 100x10      | 32        | 0.031           | 0.083            | 3.69E-01                   | 0.032089         | 9.97E+05                   | 2.57x                        |
| 100x100     | 32        | 0.305           | 0.816            | 3.74E-01                   | 0.315534         | 1.01E+06                   | 2.59x                        |
| 100x1000    | 32        | 3.052           | 8.224            | 3.71E-01                   | 3.147171         | 1.02E+06                   | 2.61x                        |
| ----------- | --------- | --------------- | ---------------- | -------------------------- | ---------------- | -------------------------- | ---------------------------- |
| 1000x10     | 32        | 0.305           | 0.824            | 3.70E-01                   | 0.03293          | 9.72E+06                   | 25.02x                       |
| 1000x100    | 32        | 3.052           | 8.254            | 3.70E-01                   | 0.324776         | 9.85E+06                   | 25.41x                       |
| 1000x1000   | 32        | 30.518          | 82.066           | 3.72E-01                   | 3.183036         | 1.01E+07                   | 25.78x                       |
| 1000x1000   | 32        | 3051.758        | \-               | \-                         | 317.945872       | 1.01E+07                   | \-                           |