load("//bazel:sxt_benchmark.bzl", "sxt_cc_benchmark")

sxt_cc_benchmark(
    name = "benchmark",
    srcs = [
        "benchmark.m.cc",
    ],
    deps = [
        "//sxt/base/container:span",
        "//sxt/curve21/type:element_p3",
        "//sxt/memory/management:managed_array",
        "//sxt/ristretto/base:byte_conversion",
        "//sxt/ristretto/type:compressed_element",
        "//sxt/seqcommit/base:indexed_exponent_sequence",
        "//sxt/seqcommit/generator:base_element",
        "//sxt/seqcommit/backend:naive_cpu_backend",
        "//sxt/seqcommit/backend:naive_gpu_backend",
        "//sxt/seqcommit/backend:pippenger_cpu_backend"
    ],
    visibility = ["//visibility:public"],
    linkopts = [
        '-x', 'cuda'
    ],
)
