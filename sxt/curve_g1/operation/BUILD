load(
    "//bazel:sxt_component.bzl",
    "sxt_cc_component",
)

sxt_cc_component(
    name = "cmov",
    impl_deps = [
        "//sxt/field12/operation:cmov",
    ],
    test_deps = [
        "//sxt/base/test:unit_test",
        "//sxt/curve_g1/constant:generator",
        "//sxt/curve_g1/constant:identity",
        "//sxt/field12/constant:one",
        "//sxt/field12/constant:zero",
    ],
    deps = [
        "//sxt/curve_g1/type:element_p2",
    ],
)
