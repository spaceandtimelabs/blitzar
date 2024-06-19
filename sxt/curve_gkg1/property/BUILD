load(
    "//bazel:sxt_build_system.bzl",
    "sxt_cc_component",
)

sxt_cc_component(
    name = "curve",
    impl_deps = [
        "//sxt/curve_gkg1/constant:b",
        "//sxt/curve_gkg1/type:element_affine",
        "//sxt/curve_gkg1/type:element_p2",
        "//sxt/field25/operation:add",
        "//sxt/field25/operation:mul",
        "//sxt/field25/operation:square",
        "//sxt/field25/operation:sub",
        "//sxt/field25/property:zero",
        "//sxt/field25/type:element",
    ],
    test_deps = [
        "//sxt/base/test:unit_test",
        "//sxt/curve_gkg1/constant:b",
        "//sxt/curve_gkg1/constant:generator",
        "//sxt/curve_gkg1/type:element_affine",
        "//sxt/curve_gkg1/type:element_p2",
        "//sxt/field25/constant:one",
        "//sxt/field25/operation:mul",
        "//sxt/field25/operation:neg",
    ],
)

sxt_cc_component(
    name = "identity",
    test_deps = [
        "//sxt/base/test:unit_test",
        "//sxt/curve_gkg1/constant:generator",
    ],
    deps = [
        "//sxt/base/macro:cuda_callable",
        "//sxt/curve_gkg1/type:element_affine",
        "//sxt/curve_gkg1/type:element_p2",
        "//sxt/field25/property:zero",
    ],
)
