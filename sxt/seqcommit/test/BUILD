load("//bazel:sxt_component.bzl", "sxt_cc_component")

sxt_cc_component(
    name = "test_pedersen",
    with_test = False,
    deps = [
        "//sxt/base/functional:function_ref",
        "//sxt/base/container:span",
        "//sxt/ristretto/type:compressed_element",
        "//sxt/seqcommit/generator:base_element",
        "//sxt/seqcommit/base:indexed_exponent_sequence",
        "//sxt/curve21/type:element_p3",
        "//sxt/ristretto/base:byte_conversion",
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
        "//sxt/base/test:unit_test",
        "//sxt/seqcommit/generator:base_element",
        "//sxt/curve21/type:element_p3",
        "//sxt/curve21/constant:zero",
        "//sxt/ristretto/base:byte_conversion",
        "//sxt/ristretto/type:compressed_element",
    ],
)
