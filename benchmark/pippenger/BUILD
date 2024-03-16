load("//bazel:sxt_benchmark.bzl", "sxt_cc_benchmark")
load(
    "//bazel:sxt_build_system.bzl",
    "sxt_cc_component",
)

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
    "//sxt/multiexp/pippenger2:multiexponentiation",
    "//sxt/memory/management:managed_array",
  ],
)
