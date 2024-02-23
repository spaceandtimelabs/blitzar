load(
    "//bazel:sxt_build_system.bzl",
    "sxt_cc_component",
)

sxt_cc_component(
    name = "curve",
    impl_deps = [
        "//sxt/curve32/type:element_p3",
        "//sxt/field32/type:element",
        "//sxt/field32/constant:d",
        "//sxt/field32/operation:add",
        "//sxt/field32/operation:mul",
        "//sxt/field32/operation:sq",
        "//sxt/field32/operation:sub",
        "//sxt/field32/property:zero",
    ],
    test_deps = [
        "//sxt/base/test:unit_test",
        "//sxt/curve32/type:literal",
    ],
)
