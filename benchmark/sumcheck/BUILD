load("//bazel:sxt_benchmark.bzl", "sxt_cc_benchmark")

sxt_cc_benchmark(
    name = "benchmark",
    srcs = [
        "benchmark.m.cc",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//sxt/base/error:panic",
        "//sxt/base/container:span",
        "//sxt/base/num:fast_random_number_generator",
        "//sxt/base/num:ceil_log2",
        "//sxt/execution/async:future",
        "//sxt/execution/schedule:scheduler",
        # "//sxt/cbindings/backend:cpu_backend",
        # "//sxt/cbindings/backend:gpu_backend",
        # "//sxt/curve21/type:element_p3",
        "//sxt/proof/sumcheck:gpu_driver",
        "//sxt/proof/sumcheck:proof_computation",
        "//sxt/proof/transcript",
        "//sxt/memory/management:managed_array",
        "//sxt/scalar25/random:element",
        "//sxt/scalar25/type:element",
        # "//sxt/multiexp/base:exponent_sequence",
        # "//sxt/ristretto/base:byte_conversion",
        # "//sxt/ristretto/type:compressed_element",
        # "//sxt/seqcommit/generator:base_element",
    ],
)
