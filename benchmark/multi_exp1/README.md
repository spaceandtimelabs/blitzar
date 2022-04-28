Benchmarking for cpu vs gpu of naive scalar multiplication

## Results

Given m and n, we compute res_j for j = 1, ..., m, where

res_j = sum_{i=1,...,n} a_{ij} * element_{ij}

where element_{ij} is an element on the elliptical curve 25519 and
a_{ij} is a random 256-bit value.

This setup uses the Azure VM [Standard_NC6s_v3](https://docs.microsoft.com/en-us/azure/virtual-machines/ncv3-series)
with a V100 GPU.

For each run, we measure the duration in milliseconds and the throughput (m * n / duration).

## Results

### CPU
```
> bazel run -c opt //benchmark/multi_exp1:benchmark -- cpu 10 100 0
===== benchmark results
m = 10
n = 100
duration (ms): 68.788
throughput: 14.5374

> bazel run -c opt //benchmark/multi_exp1:benchmark -- cpu 100 1000 0
===== benchmark results
m = 100
n = 1000
duration (ms): 6920.93
throughput: 14.4489

> bazel run -c opt //benchmark/multi_exp1:benchmark -- cpu 100 10000 0
===== benchmark results
m = 100
n = 10000
duration (ms): 69181.6
throughput: 14.4547
```

### GPU
```
> bazel run -c opt //benchmark/multi_exp1:benchmark -- gpu 100 10000 0
===== benchmark results
m = 100
n = 10000
duration (ms): 536.834
throughput: 1862.77

> bazel run -c opt //benchmark/multi_exp1:benchmark -- gpu 1000 10000 0
===== benchmark results
m = 1000
n = 10000
duration (ms): 5508.98
throughput: 1815.22
```
