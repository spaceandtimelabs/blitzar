load("//bazel:sxt_benchmark.bzl", "sxt_cc_benchmark")
load("//bazel:sxt_build_system.bzl", "sxt_cc_library")

sxt_cc_benchmark(
    name = "benchmark",
    srcs = [
        "benchmark.m.cc",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":curve_addition_bls12_381",
    ],
)

sxt_cc_library(
    name = "curve_addition_bls12_381",
    srcs = [
        "curve_addition_bls12_381.cu",
    ],
    hdrs = [
        "curve_addition_bls12_381.h",
    ],
    deps = [
        "//sxt/base/num:fast_random_number_generator",
        "//sxt/curve_g1/operation:add",
        "//sxt/curve_g1/operation:double",
        "//sxt/curve_g1/operation:neg",
        "//sxt/curve_g1/random:element_p2",
        "//sxt/curve_g1/type:element_p2",
        "//sxt/memory/management:managed_array",
        "//sxt/memory/resource:device_resource",
    ],
)
