load("//bazel:sxt_benchmark.bzl", "sxt_cc_benchmark")
load("//bazel:sxt_component.bzl", "sxt_cc_library")

sxt_cc_library(
    name = "reduce_cpu",
    srcs = [
        "reduce_cpu.cc",
    ],
    hdrs = [
        "reduce_cpu.h",
    ],
    deps = [
        "//sxt/curve21/operation:add",
        "//sxt/curve21/type:element_p3",
    ],
)

sxt_cc_library(
    name = "reduce_gpu",
    srcs = [
        "reduce_gpu.cu",
    ],
    hdrs = [
        "reduce_gpu.h",
    ],
    deps = [
        "//sxt/curve21/operation:add",
        "//sxt/curve21/type:element_p3",
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
        "//sxt/base/error:panic",
        "//sxt/curve21/type:element_p3",
    ],
)
