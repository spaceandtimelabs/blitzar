load(
    "//bazel:sxt_component.bzl",
    "sxt_cc_component",
)

sxt_cc_component(
    name = "curve",
    impl_deps = [
        "//sxt/curve_g1/constant:b",
        "//sxt/curve_g1/type:element_affine",
        "//sxt/curve_g1/type:element_p2",
        "//sxt/field12/operation:add",
        "//sxt/field12/operation:mul",
        "//sxt/field12/operation:square",
        "//sxt/field12/operation:sub",
        "//sxt/field12/property:zero",
        "//sxt/field12/type:element",
    ],
    test_deps = [
        "//sxt/base/test:unit_test",
        "//sxt/curve_g1/constant:b",
        "//sxt/curve_g1/constant:generator",
        "//sxt/curve_g1/constant:identity",
        "//sxt/curve_g1/type:element_affine",
        "//sxt/curve_g1/type:element_p2",
        "//sxt/field12/constant:one",
        "//sxt/field12/operation:mul",
        "//sxt/field12/operation:neg",
        "//sxt/field12/type:element",
    ],
)

sxt_cc_component(
    name = "identity",
    test_deps = [
        "//sxt/base/test:unit_test",
        "//sxt/curve_g1/constant:generator",
        "//sxt/curve_g1/constant:identity",
    ],
    deps = [
        "//sxt/curve_g1/type:element_affine",
        "//sxt/curve_g1/type:element_p2",
        "//sxt/field12/property:zero",
    ],
)
