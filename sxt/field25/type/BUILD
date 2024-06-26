load(
    "//bazel:sxt_build_system.bzl",
    "sxt_cc_component",
)

sxt_cc_component(
    name = "element",
    impl_deps = [
        "//sxt/field25/base:byte_conversion",
        "//sxt/field25/base:reduce",
    ],
    is_cuda = True,
    test_deps = [
        "//sxt/base/test:unit_test",
        "//sxt/field25/base:constants",
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
        "//sxt/field25/base:byte_conversion",
    ],
)
