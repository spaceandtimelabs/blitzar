load(
    "//bazel:sxt_component.bzl",
    "sxt_cc_component",
)

sxt_cc_component(
    name = "generator",
    impl_deps = [
    ],
    with_test = False,
    deps = [
        "//sxt/curve_g1/type:element_affine",
        "//sxt/field12/type:element",
    ],
)

sxt_cc_component(
    name = "identity",
    impl_deps = [
    ],
    with_test = False,
    deps = [
        "//sxt/curve_g1/type:element_affine",
        "//sxt/field12/constant:one",
        "//sxt/field12/constant:zero",
        "//sxt/field12/type:element",
    ],
)
