load("@local_config_cuda//cuda:build_defs.bzl", "cuda_library")

cc_binary(
    name = "benchmark",
    srcs = [
        "benchmark.m.cc",
    ],
    deps = [
        "//sxt/base/container:span",
        "//sxt/seqcommit/base:commitment",
        "//sxt/multiexp/base:exponent_sequence",
        "//sxt/memory/management:managed_array",
        "//sxt/base/num:fast_random_number_generator",
        "//sxt/seqcommit/naive:commitment_computation_cpu",
        "//sxt/seqcommit/naive:commitment_computation_gpu"
    ],
    visibility = ["//visibility:public"],
    linkopts = [
        '-x', 'cuda',
    ],
)
