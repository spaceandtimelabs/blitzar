load("//bazel:sxt_benchmark.bzl", "sxt_cc_benchmark")

sxt_cc_benchmark(
    name = "benchmark",
    srcs = [
        "benchmark.m.cc",
    ],
    linkopts = [
        "-x",
        "cuda",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//sxt/base/container:span",
        "//sxt/base/profile:callgrind",
        "//sxt/curve21/type:element_p3",
        "//sxt/memory/management:managed_array",
        "//sxt/multiexp/curve21:multiproduct_cpu_driver",
        "//sxt/multiexp/index:index_table",
        "//sxt/multiexp/pippenger_multiprod:multiproduct",
        "//sxt/multiexp/random:random_multiproduct_descriptor",
        "//sxt/multiexp/random:random_multiproduct_generation",
        "//sxt/multiexp/test:add_curve21_elements",
        "//sxt/ristretto/random:element",
    ],
)
