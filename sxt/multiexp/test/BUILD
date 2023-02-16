load(
    "//bazel:sxt_component.bzl",
    "sxt_cc_component",
)

sxt_cc_component(
    name = "add_ints",
    impl_deps = [
        "//sxt/base/error:panic",
    ],
    with_test = False,
    deps = [
        "//sxt/base/container:span",
    ],
)

sxt_cc_component(
    name = "curve21_arithmetic",
    impl_deps = [
        "//sxt/multiexp/base:exponent_sequence",
        "//sxt/curve21/type:element_p3",
        "//sxt/curve21/constant:zero",
        "//sxt/curve21/operation:add",
        "//sxt/curve21/operation:scalar_multiply",
        "//sxt/curve21/property:curve",
    ],
    with_test = False,
    deps = [
        "//sxt/base/container:span",
    ],
)

sxt_cc_component(
    name = "compute_uint64_muladd",
    impl_deps = [
        "//sxt/multiexp/base:exponent_sequence",
    ],
    with_test = False,
    deps = [
        "//sxt/base/container:span",
    ],
)
