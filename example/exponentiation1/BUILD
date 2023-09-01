load("@rules_cuda//cuda:defs.bzl", "cuda_library")

cuda_library(
    name = "exponentiate_gpu",
    srcs = ["exponentiate_gpu.cu"],
    hdrs = ["exponentiate_gpu.h"],
    copts = [],
    visibility = ["//visibility:public"],
    deps = [
        "//sxt/base/num:fast_random_number_generator",
        "//sxt/curve21/operation:scalar_multiply",
        "//sxt/curve21/type:element_p3",
        "//sxt/field51/type:element",
        "@local_cuda//:cuda",
    ],
)

cc_library(
    name = "exponentiate_cpu",
    srcs = ["exponentiate_cpu.cc"],
    hdrs = ["exponentiate_cpu.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//sxt/base/num:fast_random_number_generator",
        "//sxt/curve21/operation:scalar_multiply",
        "//sxt/curve21/type:element_p3",
        "//sxt/field51/type:element",
    ],
)

cc_binary(
    name = "exponentiate",
    srcs = [
        "main.cc",
    ],
    linkopts = [
        "-x",
        "cuda",
        "--cudalog",
    ],
    deps = [
        ":exponentiate_cpu",
        ":exponentiate_gpu",
        "//sxt/curve21/type:element_p3",
    ],
)
