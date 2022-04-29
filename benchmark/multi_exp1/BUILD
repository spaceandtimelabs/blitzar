load("@local_config_cuda//cuda:build_defs.bzl", "cuda_library")

cc_library(
    name = "multiply_add",
    hdrs = [
        "multiply_add.h",
    ],
    srcs = [
        "multiply_add.cc",
    ],
    copts = [
        '-x', 'cuda',
        '--device-c',
    ],
    deps = [
        "//sxt/base/macro:cuda_callable",
        "//sxt/base/num:fast_random_number_generator",
        "//sxt/curve21/operation:add",
        "//sxt/curve21/operation:scalar_multiply",
        "//sxt/curve21/random:exponent",
        "//sxt/curve21/random:ristretto_element",
    ],
)

cc_library(
    name = "multi_exp_cpu",
    hdrs = [
        "multi_exp_cpu.h",
    ],
    srcs = [
        "multi_exp_cpu.cc",
    ],
    deps = [
        ":multiply_add",
        "//sxt/curve21/constant:zero",
        "//sxt/curve21/type:element_p3",
    ],
)

cuda_library(
    name = "multi_exp_gpu",
    hdrs = [
        "multi_exp_gpu.h",
    ],
    copts = [
        '--device-c',
        '-std=c++17',
    ],
    srcs = [
        "multi_exp_gpu.cu",
    ],
    deps = [
        ":multiply_add",
        "//sxt/curve21/type:element_p3",
        "//sxt/curve21/operation:add",
        "//sxt/curve21/constant:zero",
        "//sxt/memory/management:managed_array",
        "//sxt/memory/resource:device_resource",
    ],
)

cc_binary(
    name = "benchmark",
    srcs = [
        "benchmark.m.cc",
    ],
    deps = [
        ":multi_exp_cpu",
        ":multi_exp_gpu",
        "//sxt/memory/management:managed_array",
        "//sxt/curve21/type:element_p3",
    ],
    visibility = ["//visibility:public"],
    linkopts = [
        '-x', 'cuda',
    ],
)
