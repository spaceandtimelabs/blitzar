load(
    "//bazel:sxt_build_system.bzl",
    "sxt_cc_component",
)

sxt_cc_component(
    name = "operation_adl_stub",
    with_test = False,
)

sxt_cc_component(
    name = "element",
    impl_deps = [
        "//sxt/fieldgk/base:byte_conversion",
        "//sxt/fieldgk/base:reduce",
    ],
    is_cuda = True,
    test_deps = [
        "//sxt/base/test:unit_test",
        "//sxt/fieldgk/base:constants",
    ],
    deps = [
        ":operation_adl_stub",
        "//sxt/fieldgk/base:constants",
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
        "//sxt/fieldgk/base:byte_conversion",
    ],
)
