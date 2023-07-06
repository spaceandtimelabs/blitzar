load(
    "//bazel:sxt_component.bzl",
    "sxt_cc_component",
)

sxt_cc_component(
    name = "element_affine",
    impl_deps = [
    ],
    test_deps = [
        "//sxt/base/test:unit_test",
        "//sxt/field12/constant:one",
        "//sxt/field12/constant:zero",
    ],
    deps = [
        "//sxt/field12/type:element",
    ],
)

sxt_cc_component(
    name = "element_p2",
    impl_deps = [
        "//sxt/field12/operation:mul",
        "//sxt/field12/property:zero",
    ],
    test_deps = [
        "//sxt/base/test:unit_test",
        "//sxt/field12/constant:one",
        "//sxt/field12/constant:zero",
        "//sxt/field12/operation:mul",
    ],
    deps = [
        "//sxt/field12/type:element",
    ],
)
