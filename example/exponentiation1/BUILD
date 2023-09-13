load("@local_config_cuda//cuda:build_defs.bzl", "cuda_library")
load("//bazel:sxt_component.bzl", "sxt_cc_binary", "sxt_cc_library")

sxt_cc_library(
    name = "exponentiate_gpu",
    srcs = ["exponentiate_gpu.cu"],
    hdrs = ["exponentiate_gpu.h"],
    deps = [
        "//sxt/base/num:fast_random_number_generator",
        "//sxt/curve21/operation:scalar_multiply",
        "//sxt/curve21/type:element_p3",
        "//sxt/field51/type:element",
        "@local_config_cuda//cuda",
    ],
)

sxt_cc_library(
    name = "exponentiate_cpu",
    srcs = ["exponentiate_cpu.cc"],
    hdrs = ["exponentiate_cpu.h"],
    deps = [
        "//sxt/base/num:fast_random_number_generator",
        "//sxt/curve21/operation:scalar_multiply",
        "//sxt/curve21/type:element_p3",
        "//sxt/field51/type:element",
    ],
)

sxt_cc_binary(
    name = "exponentiate",
    srcs = [
        "main.cc",
    ],
    deps = [
        ":exponentiate_cpu",
        ":exponentiate_gpu",
        "//sxt/curve21/type:element_p3",
    ],
)
