load(
    "//bazel:sxt_component.bzl",
    "sxt_cc_component",
)

sxt_cc_component(
    name = "curve",
    impl_deps = [
        "//sxt/curve21/type:element_p3",
        "//sxt/field51/type:element",
        "//sxt/field51/constant:d",
        "//sxt/field51/operation:add",
        "//sxt/field51/operation:mul",
        "//sxt/field51/operation:square",
        "//sxt/field51/operation:sub",
        "//sxt/field51/property:zero",
    ],
    test_deps = [
        "//sxt/curve21/type:element_p3",
        "//sxt/base/test:unit_test",
    ],
)
