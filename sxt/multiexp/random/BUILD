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
    impl_deps = [
        ":random_multiexponentiation_descriptor",
        "//sxt/multiexp/base:exponent_sequence",
    ],
    with_test = False,
    deps = [
        "//sxt/base/container:span",
        "//sxt/base/memory:alloc",
    ],
)

sxt_cc_component(
    name = "random_multiproduct_descriptor",
    with_test = False,
)

sxt_cc_component(
    name = "random_multiproduct_generation",
    impl_deps = [
        ":random_multiproduct_descriptor",
        "//sxt/multiexp/index:index_table",
        "//sxt/multiexp/index:reindex",
        "//sxt/memory/management:managed_array",
    ],
    with_test = False,
    deps = [
        "//sxt/base/container:span",
        "//sxt/memory/management:managed_array_fwd",
    ],
)
