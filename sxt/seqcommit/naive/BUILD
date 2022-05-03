load(
    "//bazel:sxt_component.bzl",
    "sxt_cc_component",
)

sxt_cc_component(
    name = "commitment_computation",
    deps = [
        "//sxt/base/container:span",
    ],
    impl_deps = [
        "//sxt/seqcommit/base:commitment",
        "//sxt/seqcommit/base:base_element",
        "//sxt/multiexp/base:exponent_sequence",
    ],
    test_deps = [
        "//sxt/base/test:unit_test",
    ],
)
