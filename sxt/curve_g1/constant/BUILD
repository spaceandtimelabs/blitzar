load(
    "//bazel:sxt_component.bzl",
    "sxt_cc_component",
)

sxt_cc_component(
    name = "b",
    test_deps = [
        "//sxt/base/test:unit_test",
        "//sxt/field12/base:montgomery",
    ],
    deps = [
        "//sxt/field12/type:element",
    ],
)

sxt_cc_component(
    name = "beta",
    test_deps = [
        "//sxt/base/test:unit_test",
        "//sxt/field12/base:byte_conversion",
        "//sxt/field12/constant:one",
        "//sxt/field12/operation:mul",
    ],
    deps = [
        "//sxt/field12/type:element",
    ],
)

sxt_cc_component(
    name = "generator",
    with_test = False,
    deps = [
        "//sxt/curve_g1/type:element_affine",
        "//sxt/curve_g1/type:element_p2",
        "//sxt/field12/constant:one",
        "//sxt/field12/type:element",
    ],
)

sxt_cc_component(
    name = "identity",
    with_test = False,
    deps = [
        "//sxt/curve_g1/type:element_affine",
        "//sxt/curve_g1/type:element_p2",
        "//sxt/field12/constant:one",
        "//sxt/field12/constant:zero",
        "//sxt/field12/type:element",
    ],
)
