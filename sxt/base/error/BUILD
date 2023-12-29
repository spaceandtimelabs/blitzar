load(
    "//bazel:sxt_build_system.bzl",
    "sxt_cc_component",
)

sxt_cc_component(
    name = "stacktrace",
    impl_deps = [
        "@com_github_ianlancetaylor_libbacktrace//:libbacktrace",
        "@boost//:stacktrace",
    ],
    with_test = False,
)

sxt_cc_component(
    name = "panic",
    impl_deps = [
        ":stacktrace",
    ],
    with_test = False,
)

sxt_cc_component(
    name = "assert",
    test_deps = [
        "//sxt/base/test:unit_test",
    ],
    deps = [
        ":stacktrace",
    ],
)
