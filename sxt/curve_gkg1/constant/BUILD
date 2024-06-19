load(
    "//bazel:sxt_build_system.bzl",
    "sxt_cc_component",
)

sxt_cc_component(
    name = "b",
    test_deps = [
        "//sxt/base/test:unit_test",
        "//sxt/field25/base:montgomery",
    ],
    deps = [
        "//sxt/field25/type:element",
    ],
)

sxt_cc_component(
    name = "generator",
    test_deps = [
        "//sxt/base/test:unit_test",
        "//sxt/field25/base:montgomery",
    ],
    deps = [
        "//sxt/curve_gkg1/type:element_affine",
        "//sxt/curve_gkg1/type:element_p2",
        "//sxt/field25/constant:one",
        "//sxt/field25/type:element",
    ],
)
