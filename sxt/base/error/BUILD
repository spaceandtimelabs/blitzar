load(
    "//bazel:sxt_build_system.bzl",
    "sxt_cc_component",
)

sxt_cc_component(
    name = "stacktrace",
    with_test = False,
    impl_deps = [
      "@com_github_ianlancetaylor_libbacktrace//:libbacktrace",
      "@boost//:stacktrace",
    ],
)

sxt_cc_component(
    name = "panic",
    with_test = False,
)

sxt_cc_component(
    name = "assert",
    deps = [
      ":stacktrace",
    ],
    test_deps = [
        "//sxt/base/test:unit_test",
    ],
)
