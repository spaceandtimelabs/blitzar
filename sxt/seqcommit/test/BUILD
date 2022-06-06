load("//bazel:sxt_component.bzl", "sxt_cc_component")

sxt_cc_component(
    name = "test_pedersen",
    with_test = False,
    deps = [
        "//sxt/base/functional:function_ref",
        "//sxt/base/container:span",
        "//sxt/seqcommit/base:commitment",
        "//sxt/seqcommit/generator:base_element",
        "//sxt/multiexp/base:exponent_sequence",
        "//sxt/curve21/type:element_p3",
        "//sxt/curve21/ristretto:byte_conversion",
        "//sxt/curve21/operation:scalar_multiply",
        "//sxt/curve21/operation:add",
        "//sxt/base/test:unit_test",
    ],
)

sxt_cc_component(
    name = "test_generators",
    with_test = False,
    deps = [
        "//sxt/base/functional:function_ref",
        "//sxt/base/container:span",
        "//sxt/seqcommit/generator:base_element",
        "//sxt/curve21/type:element_p3",
        "//sxt/curve21/constant:zero",
        "//sxt/base/test:unit_test",
        "//sxt/seqcommit/base:commitment",
        "//sxt/curve21/ristretto:byte_conversion"
    ],
)
