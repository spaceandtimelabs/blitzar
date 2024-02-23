load(
    "//bazel:sxt_build_system.bzl",
    "sxt_cc_component",
)

sxt_cc_component(
    name = "element",
    impl_deps = [
        "//sxt/field32/base:byte_conversion",
        "//sxt/field32/base:reduce",
    ],
    test_deps = [
        ":literal",
        "//sxt/base/test:unit_test",
        "//sxt/field32/base:byte_conversion",
        "//sxt/field51/base:byte_conversion",
        "//sxt/field51/type:element",
    ],
    deps = [
        "//sxt/base/macro:cuda_callable",
    ],
)

sxt_cc_component(
    name = "literal",
    test_deps = [
        "//sxt/base/test:unit_test",
    ],
    deps = [
        ":element",
        "//sxt/base/type:literal",
        "//sxt/field32/base:byte_conversion",
    ],
)
