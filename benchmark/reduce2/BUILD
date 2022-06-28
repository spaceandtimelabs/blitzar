load("//bazel:sxt_benchmark.bzl", "sxt_cc_benchmark")
load("@local_config_cuda//cuda:build_defs.bzl", "cuda_library")

cc_library(
    name = "reduce_cpu",
    hdrs = [
        "reduce_cpu.h",
    ],
    srcs = [
        "reduce_cpu.cc",
    ],
    deps = [
        "//sxt/curve21/operation:add",
        "//sxt/curve21/type:element_p3",
        "//sxt/curve21/constant:zero",
    ],
)

cuda_library(
    name = "reduce_gpu",
    hdrs = [
        "reduce_gpu.h",
    ],
    copts = [
        '--device-c',
    ],
    srcs = [
        "reduce_gpu.cu",
    ],
    deps = [
        "//sxt/curve21/operation:add",
        "//sxt/curve21/type:element_p3",
        "//sxt/curve21/constant:zero",
    ],
)

sxt_cc_benchmark(
    name = "benchmark",
    srcs = [
        "benchmark.m.cc",
    ],
    deps = [
        ":reduce_cpu",
        ":reduce_gpu",
        "//sxt/curve21/type:element_p3",
    ],
    visibility = ["//visibility:public"],
    linkopts = [
        '-x', 'cuda',
    ],
)
