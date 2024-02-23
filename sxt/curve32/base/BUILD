load(
    "//bazel:sxt_build_system.bzl",
    "sxt_cc_component",
)

sxt_cc_component(
    name = "elligate",
    impl_deps = [
        "//sxt/base/error:panic",
        "//sxt/field32/constant:zero",
        "//sxt/field32/type:element",
        "//sxt/field32/operation:sq",
        "//sxt/field32/operation:sqrt",
        "//sxt/field32/operation:invert",
        "//sxt/field32/operation:mul",
        "//sxt/field32/operation:add",
        "//sxt/field32/operation:neg",
        "//sxt/field32/operation:cmov",
        "//sxt/field32/operation:notsquare",
    ],
    test_deps = [
        "//sxt/base/test:unit_test",
        "//sxt/field32/type:element",
    ],
)

sxt_cc_component(
    name = "mont_ed_conversion",
    impl_deps = [
        "//sxt/field32/constant:one",
        "//sxt/field32/operation:add",
        "//sxt/field32/operation:cmov",
        "//sxt/field32/operation:mul",
        "//sxt/field32/operation:invert",
        "//sxt/field32/operation:sub",
        "//sxt/field32/property:zero",
    ],
    with_test = False,
)
