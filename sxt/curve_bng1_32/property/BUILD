load(
    "//bazel:sxt_build_system.bzl",
    "sxt_cc_component",
)

sxt_cc_component(
    name = "curve",
    impl_deps = [
        "//sxt/curve_bng1_32/constant:b",
        "//sxt/curve_bng1_32/type:element_affine",
        "//sxt/curve_bng1_32/type:element_p2",
        "//sxt/field32/operation:add",
        "//sxt/field32/operation:mul",
        "//sxt/field32/operation:square",
        "//sxt/field32/operation:sub",
        "//sxt/field32/property:zero",
        "//sxt/field32/type:element",
    ],
    test_deps = [
        "//sxt/base/test:unit_test",
        "//sxt/curve_bng1_32/constant:b",
        "//sxt/curve_bng1_32/constant:generator",
        "//sxt/curve_bng1_32/type:element_affine",
        "//sxt/curve_bng1_32/type:element_p2",
        "//sxt/field32/constant:one",
        "//sxt/field32/operation:mul",
        "//sxt/field32/operation:neg",
    ],
)

sxt_cc_component(
    name = "identity",
    test_deps = [
        "//sxt/base/test:unit_test",
        "//sxt/curve_bng1_32/constant:generator",
    ],
    deps = [
        "//sxt/base/macro:cuda_callable",
        "//sxt/curve_bng1_32/type:element_affine",
        "//sxt/curve_bng1_32/type:element_p2",
        "//sxt/field32/property:zero",
    ],
)
