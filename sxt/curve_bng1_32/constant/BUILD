load(
    "//bazel:sxt_build_system.bzl",
    "sxt_cc_component",
)

sxt_cc_component(
    name = "b",
    test_deps = [
        "//sxt/base/test:unit_test",
        "//sxt/field32/base:montgomery",
    ],
    deps = [
        "//sxt/field32/type:element",
    ],
)

sxt_cc_component(
    name = "generator",
    test_deps = [
        "//sxt/base/test:unit_test",
        "//sxt/field32/base:montgomery",
    ],
    deps = [
        "//sxt/curve_bng1_32/type:element_affine",
        "//sxt/curve_bng1_32/type:element_p2",
        "//sxt/field32/constant:one",
        "//sxt/field32/type:element",
    ],
)
