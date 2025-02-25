load("//bazel:sxt_benchmark.bzl", "sxt_cc_benchmark")

sxt_cc_benchmark(
    name = "benchmark",
    srcs = [
        "benchmark.m.cc",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//sxt/base/container:span",
        "//sxt/base/error:panic",
        "//sxt/base/num:ceil_log2",
        "//sxt/base/num:fast_random_number_generator",
        "//sxt/execution/async:future",
        "//sxt/execution/schedule:scheduler",
        "//sxt/memory/management:managed_array",
        "//sxt/proof/sumcheck2:gpu_driver",
        "//sxt/proof/sumcheck2:proof_computation",
        "//sxt/proof/sumcheck2:reference_transcript",
        "//sxt/proof/transcript",
        "//sxt/scalar25/random:element",
        "//sxt/scalar25/realization:field",
    ],
)
