load(
    "//bazel:sxt_build_system.bzl",
    "sxt_cc_component",
)

sxt_cc_component(
    name = "element",
    impl_deps = [
        "//sxt/field12/base:byte_conversion",
        "//sxt/field12/base:reduce",
    ],
    is_cuda = True,
    test_deps = [
        "//sxt/base/test:unit_test",
        "//sxt/field12/base:constants",
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
        "//sxt/field12/base:byte_conversion",
    ],
)
