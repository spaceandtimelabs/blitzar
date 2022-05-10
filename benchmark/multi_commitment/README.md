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

| Cols x Rows | Word Size | Table Size (MB) | CPU Duration (s) | CPU Throughput<br>(Exponentiations / s) | GPU Duration (s) | GPU Throughput<br>(Exponentiations / s) | Duration Speedup (GPU / CPU) |
| ----------- | --------- | --------------- | ---------------- | --------------------------------------- | ---------------- | --------------------------------------- | ---------------------------- |
| 10x10       | 4         | 0.000           | 0.009            | 1.16E+04                                | 0.006266         | 1.60E+04                                | 1.38x                        |
| 10x100      | 4         | 0.004           | 0.081            | 1.23E+04                                | 0.007669         | 1.30E+05                                | 10.60x                       |
| 10x1000     | 4         | 0.038           | 0.813            | 1.23E+04                                | 0.009351         | 1.07E+06                                | 86.91x                       |
| 100x10      | 4         | 0.004           | 0.082            | 1.22E+04                                | 0.034208         | 2.92E+04                                | 2.40x                        |
| 100x100     | 4         | 0.038           | 0.815            | 1.23E+04                                | 0.043283         | 2.31E+05                                | 18.83x                       |
| 100x1000    | 4         | 0.381           | 8.167            | 1.22E+04                                | 0.057387         | 1.74E+06                                | 142.31x                      |
| 1000x10     | 4         | 0.038           | 0.821            | 1.22E+04                                | 0.30366          | 3.29E+04                                | 2.71x                        |
| 1000x100    | 4         | 0.381           | 8.164            | 1.22E+04                                | 0.397974         | 2.51E+05                                | 20.51x                       |
| 1000x1000   | 4         | 3.815           | 81.722           | 1.22E+04                                | 0.541354         | 1.85E+06                                | 150.96x                      |
| 1000x100000 | 4         | 381.470         | \-               | \-                                      | 54.568354        | 1.83E+06                                | \-                           |