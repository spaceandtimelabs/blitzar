load(
    "//bazel:sxt_build_system.bzl",
    "sxt_cc_component",
)

sxt_cc_component(
    name = "panic",
    with_test = False,
)

sxt_cc_component(
    name = "assert",
    test_deps = [
        "//sxt/base/test:unit_test",
    ],
)
