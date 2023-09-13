load(
    "//bazel:sxt_build_system.bzl",
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
        "//sxt/field51/operation:sq",
        "//sxt/field51/operation:sub",
        "//sxt/field51/property:zero",
    ],
    test_deps = [
        "//sxt/curve21/type:literal",
        "//sxt/base/test:unit_test",
    ],
)
