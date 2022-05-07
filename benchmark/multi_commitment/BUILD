load("@local_config_cuda//cuda:build_defs.bzl", "cuda_library")

cc_library(
    name = "multi_commitment_cpu",
    hdrs = [
        "multi_commitment_cpu.h",
    ],
    srcs = [
        "multi_commitment_cpu.cc",
    ],
    deps = [
        "//sxt/base/container:span",
        "//sxt/seqcommit/base:commitment",
        "//sxt/multiexp/base:exponent_sequence",
        "//sxt/memory/management:managed_array",
        "//sxt/seqcommit/naive:commitment_computation",
    ],
)

cuda_library(
    name = "multi_commitment_gpu",
    hdrs = [
        "multi_commitment_gpu.h",
    ],
    copts = [
        '--device-c',
    ],
    srcs = [
        "multi_commitment_gpu.cu",
    ],
    deps = [
        "//sxt/base/container:span",
        "//sxt/seqcommit/base:commitment",
        "//sxt/multiexp/base:exponent_sequence",
        "//sxt/memory/management:managed_array",
        "//sxt/memory/resource:device_resource",
        "//sxt/seqcommit/naive:commitment_computation",
    ],
)


cc_binary(
    name = "benchmark",
    srcs = [
        "benchmark.m.cc",
    ],
    deps = [
        ":multi_commitment_cpu",
        ":multi_commitment_gpu",
        "//sxt/seqcommit/base:commitment",
        "//sxt/memory/management:managed_array",
        "//sxt/base/num:fast_random_number_generator"
    ],
    visibility = ["//visibility:public"],
    linkopts = [
        '-x', 'cuda'
    ],
)
