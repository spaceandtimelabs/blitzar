load("//bazel:sxt_benchmark.bzl", "sxt_cc_benchmark")

sxt_cc_benchmark(
    name = "benchmark",
    srcs = [
        "benchmark.m.cc",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//sxt/base/container:span",
        "//sxt/base/profile:callgrind",
        "//sxt/curve21/operation:add",
        "//sxt/curve21/operation:double",
        "//sxt/curve21/operation:neg",
        "//sxt/curve21/type:element_p3",
        "//sxt/memory/management:managed_array",
        "//sxt/multiexp/curve:multiproduct_cpu_driver",
        "//sxt/multiexp/index:index_table",
        "//sxt/multiexp/pippenger_multiprod:multiproduct",
        "//sxt/multiexp/random:random_multiproduct_descriptor",
        "//sxt/multiexp/random:random_multiproduct_generation",
        "//sxt/multiexp/test:curve21_arithmetic",
        "//sxt/ristretto/random:element",
    ],
)
