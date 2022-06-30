load(
    "//bazel:sxt_component.bzl",
    "sxt_cc_component",
)

sxt_cc_component(
    name = "random_multiexponentiation_descriptor",
    with_test = False,
)

sxt_cc_component(
    name = "random_multiexponentiation_generation",
    deps = [
        ":random_multiexponentiation_descriptor",
        "//sxt/base/container:span",
        "//sxt/base/type:polymorphic_allocator"
    ],
    impl_deps = [
        ":random_multiexponentiation_descriptor",
        "//sxt/multiexp/base:exponent_sequence"
    ],
    with_test = False,
)
