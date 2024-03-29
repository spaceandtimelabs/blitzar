load("//bazel:sxt_benchmark.bzl", "sxt_cc_benchmark")

sxt_cc_benchmark(
    name = "benchmark",
    srcs = [
        "main.cc",
    ],
    deps = [
        "//sxt/curve21/operation:add",
        "//sxt/curve21/operation:double",
        "//sxt/curve21/operation:neg",
        "//sxt/curve21/type:element_p3",
        "//sxt/curve21/type:literal",
        "//sxt/execution/schedule:scheduler",
        "//sxt/memory/management:managed_array",
        "//sxt/multiexp/pippenger2:multiexponentiation",
    ],
)
