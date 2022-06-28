load("//bazel:sxt_benchmark.bzl", "sxt_cc_benchmark")

sxt_cc_benchmark(
    name = "benchmark",
    srcs = [
        "benchmark.m.cc",
    ],
    deps = [
        "//sxt/base/container:span",
        "//sxt/curve21/type:element_p3",
        "//sxt/seqcommit/base:commitment",
        "//sxt/seqcommit/base:indexed_exponent_sequence",
        "//sxt/memory/management:managed_array",
        "//sxt/seqcommit/generator:base_element",
        "//sxt/curve21/ristretto:byte_conversion",
        "//sxt/seqcommit/cbindings:pedersen_cpu_backend",
        "//sxt/seqcommit/cbindings:pedersen_gpu_backend"
    ],
    visibility = ["//visibility:public"],
    linkopts = [
        '-x', 'cuda'
    ],
)
