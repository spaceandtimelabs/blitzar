load("//bazel:sxt_component.bzl", "sxt_cc_component")
load("@local_config_cuda//cuda:build_defs.bzl", "cuda_library")

sxt_cc_component(
    name = "reduce_exponent",
    is_cuda = True,
    impl_deps = [
        "//sxt/base/bit:load",
    ],
    test_deps = [
        "//sxt/base/test:unit_test",
    ],
)

sxt_cc_component(
    name = "commitment_computation_cpu",
    deps = [
        "//sxt/base/container:span",
    ],
    impl_deps = [
        ":reduce_exponent",
        "//sxt/seqcommit/base:commitment",
        "//sxt/seqcommit/base:base_element",
        "//sxt/multiexp/base:exponent_sequence",
        "//sxt/curve21/type:element_p3",
        "//sxt/curve21/ristretto:byte_conversion",
        "//sxt/curve21/operation:scalar_multiply",
        "//sxt/curve21/operation:add",
    ],
    test_deps = [
        "//sxt/base/test:unit_test",
    ],
)

cuda_library(
    name = "commitment_computation_gpu",
    hdrs = [
        "commitment_computation_gpu.h",
    ],
    copts = [
        '--device-c', '-fopenmp'
    ],
    srcs = [
        "commitment_computation_gpu.cu",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":reduce_exponent",
        "//sxt/curve21/constant:zero",
        "//sxt/base/container:span",
        "//sxt/seqcommit/base:commitment",
        "//sxt/seqcommit/base:base_element",
        "//sxt/multiexp/base:exponent_sequence",
        "//sxt/curve21/type:element_p3",
        "//sxt/curve21/ristretto:byte_conversion",
        "//sxt/curve21/operation:scalar_multiply",
        "//sxt/curve21/operation:add",
        "//sxt/memory/management:managed_array",
        "//sxt/memory/resource:device_resource",
    ],
)