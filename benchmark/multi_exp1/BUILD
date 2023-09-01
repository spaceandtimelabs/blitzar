load("//bazel:sxt_benchmark.bzl", "sxt_cc_benchmark")
load("@rules_cuda//cuda:defs.bzl", "cuda_library")

cc_library(
    name = "multiply_add",
    srcs = [
        "multiply_add.cc",
    ],
    hdrs = [
        "multiply_add.h",
    ],
    copts = [
        "-x",
        "cuda",
    ],
    deps = [
        "//sxt/base/macro:cuda_callable",
        "//sxt/base/num:fast_random_number_generator",
        "//sxt/curve21/operation:add",
        "//sxt/curve21/operation:scalar_multiply",
        "//sxt/curve21/random:exponent",
        "//sxt/ristretto/random:element",
    ],
)

cc_library(
    name = "multi_exp_cpu",
    srcs = [
        "multi_exp_cpu.cc",
    ],
    hdrs = [
        "multi_exp_cpu.h",
    ],
    deps = [
        ":multiply_add",
        "//sxt/curve21/type:element_p3",
    ],
)

cuda_library(
    name = "multi_exp_gpu",
    srcs = [
        "multi_exp_gpu.cu",
    ],
    hdrs = [
        "multi_exp_gpu.h",
    ],
    copts = [
    ],
    deps = [
        ":multiply_add",
        "//sxt/curve21/operation:add",
        "//sxt/curve21/type:element_p3",
        "//sxt/memory/management:managed_array",
        "//sxt/memory/resource:device_resource",
    ],
)

sxt_cc_benchmark(
    name = "benchmark",
    srcs = [
        "benchmark.m.cc",
    ],
    linkopts = [
        "-x",
        "cuda",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":multi_exp_cpu",
        ":multi_exp_gpu",
        "//sxt/base/error:panic",
        "//sxt/curve21/type:element_p3",
        "//sxt/memory/management:managed_array",
    ],
)
