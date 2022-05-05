load(
    "//bazel:sxt_component.bzl",
    "sxt_cc_component",
)

sxt_cc_component(
    name = "reduce_exponent",
    impl_deps = [
        "//sxt/base/bit:load",
    ],
    test_deps = [
        "//sxt/base/test:unit_test",
    ],
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
        "//sxt/curve21/type:element_p3",
        "//sxt/curve21/ristretto:byte_conversion",
        "//sxt/curve21/operation:scalar_multiply",
        "//sxt/curve21/operation:add",
        "//sxt/seqcommit/naive:reduce_exponent",
    ],
    test_deps = [
        "//sxt/base/test:unit_test",
    ],
)