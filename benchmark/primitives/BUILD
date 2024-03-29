load("//bazel:sxt_benchmark.bzl", "sxt_cc_benchmark")
load("//bazel:sxt_build_system.bzl", "sxt_cc_library")

sxt_cc_benchmark(
    name = "benchmark",
    srcs = [
        "benchmark.m.cc",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":curve_ops_bls12_381",
        ":curve_ops_bn254",
        ":field_ops_bls12_381",
        ":field_ops_bn254",
    ],
)

sxt_cc_library(
    name = "curve_ops_bls12_381",
    srcs = [
        "curve_ops_bls12_381.cu",
    ],
    hdrs = [
        "curve_ops_bls12_381.h",
    ],
    deps = [
        "//sxt/base/num:divide_up",
        "//sxt/base/num:fast_random_number_generator",
        "//sxt/curve_g1/operation:add",
        "//sxt/curve_g1/random:element_p2",
        "//sxt/curve_g1/type:element_p2",
        "//sxt/memory/management:managed_array",
        "//sxt/memory/resource:device_resource",
    ],
)

sxt_cc_library(
    name = "curve_ops_bn254",
    srcs = [
        "curve_ops_bn254.cu",
    ],
    hdrs = [
        "curve_ops_bn254.h",
    ],
    deps = [
        "//sxt/base/num:divide_up",
        "//sxt/base/num:fast_random_number_generator",
        "//sxt/curve_bng1/operation:add",
        "//sxt/curve_bng1/random:element_p2",
        "//sxt/curve_bng1/type:element_p2",
        "//sxt/memory/management:managed_array",
        "//sxt/memory/resource:device_resource",
    ],
)

sxt_cc_library(
    name = "field_ops_bls12_381",
    srcs = [
        "field_ops_bls12_381.cu",
    ],
    hdrs = [
        "field_ops_bls12_381.h",
    ],
    deps = [
        ":stats",
        "//sxt/base/num:divide_up",
        "//sxt/base/num:fast_random_number_generator",
        "//sxt/field12/operation:add",
        "//sxt/field12/operation:mul",
        "//sxt/field12/random:element",
        "//sxt/field12/type:element",
        "//sxt/memory/management:managed_array",
        "//sxt/memory/resource:device_resource",
    ],
)

sxt_cc_library(
    name = "field_ops_bn254",
    srcs = [
        "field_ops_bn254.cu",
    ],
    hdrs = [
        "field_ops_bn254.h",
    ],
    deps = [
        ":stats",
        "//sxt/base/num:divide_up",
        "//sxt/base/num:fast_random_number_generator",
        "//sxt/field25/operation:add",
        "//sxt/field25/operation:mul",
        "//sxt/field25/random:element",
        "//sxt/field25/type:element",
        "//sxt/memory/management:managed_array",
        "//sxt/memory/resource:device_resource",
    ],
)

sxt_cc_library(
    name = "stats",
    srcs = [
        "stats.cc",
    ],
    hdrs = [
        "stats.h",
    ],
)
