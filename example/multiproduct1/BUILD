load("//bazel:sxt_benchmark.bzl", "sxt_cc_benchmark")

sxt_cc_benchmark(
    name = "t",
    srcs = [
        "main.cc",
    ],
    linkopts = [
        "-x",
        "cuda",
    ],
    deps = [
        "//sxt/base/profile:callgrind",
        "//sxt/curve21/type:element_p3",
        "//sxt/memory/management:managed_array",
        "//sxt/multiexp/curve21:multiproduct_cpu_driver",
        "//sxt/multiexp/index:index_table",
        "//sxt/multiexp/pippenger_multiprod:multiproduct",
        "//sxt/multiexp/test:generate_curve21_elements",
    ],
)
