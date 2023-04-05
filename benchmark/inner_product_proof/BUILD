load(
    "//bazel:sxt_component.bzl",
    "sxt_cc_component",
)
load("//bazel:sxt_benchmark.bzl", "sxt_cc_benchmark")

sxt_cc_component(
    name = "params",
    impl_deps = [
        "//sxt/base/error:assert",
        "//sxt/base/error:panic",
        "//sxt/cbindings/backend:cpu_backend",
        "//sxt/cbindings/backend:gpu_backend",
    ],
    with_test = False,
    deps = [
        "//sxt/cbindings/backend:computational_backend",
    ],
)

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
        ":params",
        "//sxt/base/container:span",
        "//sxt/base/num:ceil_log2",
        "//sxt/base/num:fast_random_number_generator",
        "//sxt/base/profile:callgrind",
        "//sxt/curve21/operation:overload",
        "//sxt/proof/inner_product:proof_descriptor",
        "//sxt/proof/inner_product:random_product_generation",
        "//sxt/proof/transcript",
        "//sxt/ristretto/type:compressed_element",
        "//sxt/scalar25/operation:overload",
        "//sxt/scalar25/type:element",
    ],
)
