# Benchmarking for cpu vs gpu of naive scalar multiplication

## Results

Given m and n, we compute res_j for j = 1, ..., m, where

res_j = sum_{i=1,...,n} element_{ij}

where element_{ij} is an element on the elliptical curve 25519.

This setup uses the Azure VM [Standard_NC6s_v3](https://docs.microsoft.com/en-us/azure/virtual-machines/ncv3-series)
with a V100 GPU.

For each run, we measure the duration in milliseconds and the throughput (m * n / duration).

## Results

### CPU
```
> bazel run -c opt //benchmark/reduce2:benchmark -- cpu 10 1000 0
===== benchmark results
m = 10
n = 1000
duration (ms): 2.21
throughput: 4524.89

> bazel run -c opt //benchmark/reduce2:benchmark -- cpu 100 10000 0
===== benchmark results
m = 100
n = 10000
duration (ms): 221.65
throughput: 4511.62
```

### GPU
```
> bazel run -c opt //benchmark/reduce2:benchmark -- gpu 100 10000 0
===== benchmark results
m = 100
n = 10000
duration (ms): 1.326
throughput: 754148

> bazel run -c opt //benchmark/reduce2:benchmark -- gpu 1000 10000 0
===== benchmark results
m = 1000
n = 10000
duration (ms): 8.81
throughput: 1.13507e+06

> bazel run -c opt //benchmark/reduce2:benchmark -- gpu 1000 100000 0
===== benchmark results
m = 1000
n = 100000
duration (ms): 85.795
throughput: 1.16557e+06

> bazel run -c opt //benchmark/reduce2:benchmark -- gpu 1000 1000000 0
===== benchmark results
m = 1000
n = 1000000
duration (ms): 730.038
throughput: 1.36979e+06

> bazel run -c opt //benchmark/reduce2:benchmark -- gpu 2000 1000000 0
===== benchmark results
m = 2000
n = 1000000
duration (ms): 1340.64
throughput: 1.49183e+06
```
