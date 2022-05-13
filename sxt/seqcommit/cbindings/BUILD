load(
    "//bazel:sxt_component.bzl",
    "sxt_cc_component",
)

sxt_cc_component(
    name = "pedersen",
    impl_deps = [
        "//sxt/base/container:span",
        "//sxt/seqcommit/base:commitment",
        "//sxt/multiexp/base:exponent_sequence",
        "//sxt/seqcommit/naive:commitment_computation_cpu",
        "//sxt/seqcommit/naive:commitment_computation_gpu",
    ],
    test_deps = [
        "//sxt/base/test:unit_test",
    ],
)
