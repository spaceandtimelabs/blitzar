load(
    "//bazel:sxt_build_system.bzl",
    "sxt_cc_component",
)

sxt_cc_component(
    name = "elligate",
    impl_deps = [
        "//sxt/base/error:panic",
        "//sxt/field51/constant:zero",
        "//sxt/field51/type:element",
        "//sxt/field51/operation:sq",
        "//sxt/field51/operation:sqrt",
        "//sxt/field51/operation:invert",
        "//sxt/field51/operation:mul",
        "//sxt/field51/operation:add",
        "//sxt/field51/operation:neg",
        "//sxt/field51/operation:cmov",
        "//sxt/field51/operation:notsquare",
    ],
    test_deps = [
        "//sxt/field51/type:element",
        "//sxt/base/test:unit_test",
    ],
)

sxt_cc_component(
    name = "mont_ed_conversion",
    impl_deps = [
        "//sxt/field51/constant:one",
        "//sxt/field51/operation:add",
        "//sxt/field51/operation:cmov",
        "//sxt/field51/operation:mul",
        "//sxt/field51/operation:invert",
        "//sxt/field51/operation:sub",
        "//sxt/field51/property:zero",
    ],
    with_test = False,
)
