load("//bazel:sxt_benchmark.bzl", "sxt_cc_benchmark")

sxt_cc_benchmark(
    name = "benchmark",
    srcs = [
        "benchmark.m.cc",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//sxt/curve21/operation:add",
        "//sxt/curve21/operation:double",
        "//sxt/curve21/operation:neg",
        "//sxt/curve21/type:element_p3",
        "//sxt/execution/schedule:scheduler",
        "//sxt/memory/resource:pinned_resource",
        "//sxt/multiexp/pippenger2:in_memory_partition_table_accessor_utility",
        "//sxt/multiexp/pippenger2:multiexponentiation",
        "//sxt/ristretto/operation:compression",
        "//sxt/ristretto/type:compressed_element",
        "//sxt/seqcommit/generator:base_element",
    ],
)
