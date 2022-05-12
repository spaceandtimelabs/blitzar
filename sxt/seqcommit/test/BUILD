load("//bazel:sxt_component.bzl", "sxt_cc_component")

sxt_cc_component(
    name = "test_commitment_computation",
    with_test = False,
    deps = [
        "//sxt/base/functional:function_ref",
        "//sxt/base/container:span",
        "//sxt/seqcommit/base:commitment",
        "//sxt/seqcommit/base:base_element",
        "//sxt/multiexp/base:exponent_sequence",
        "//sxt/curve21/type:element_p3",
        "//sxt/curve21/ristretto:byte_conversion",
        "//sxt/curve21/operation:scalar_multiply",
        "//sxt/curve21/operation:add",
        "//sxt/base/test:unit_test",
    ],
)
