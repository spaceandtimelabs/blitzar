load("//bazel:sxt_benchmark.bzl", "sxt_cc_benchmark")
load("@local_config_cuda//cuda:build_defs.bzl", "cuda_library")

cc_library(
    name = "reduce_cpu",
    srcs = [
        "reduce_cpu.cc",
    ],
    hdrs = [
        "reduce_cpu.h",
    ],
    copts = [
        "-fconcepts",
        "-std=c++20",
    ],
    deps = [
        "//sxt/curve21/constant:zero",
        "//sxt/curve21/operation:add",
        "//sxt/curve21/type:element_p3",
    ],
)

cuda_library(
    name = "reduce_gpu",
    srcs = [
        "reduce_gpu.cu",
    ],
    hdrs = [
        "reduce_gpu.h",
    ],
    copts = [
        "--device-c",
    ],
    deps = [
        "//sxt/curve21/constant:zero",
        "//sxt/curve21/operation:add",
        "//sxt/curve21/type:element_p3",
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
        ":reduce_cpu",
        ":reduce_gpu",
        "//sxt/base/error:panic",
        "//sxt/curve21/type:element_p3",
    ],
)
