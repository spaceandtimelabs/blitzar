load(
    "//bazel:sxt_build_system.bzl",
    "sxt_cc_component",
)

sxt_cc_component(
    name = "file_io",
    test_deps = [
        "//sxt/base/test:unit_test",
    ],
    impl_deps = [
      "//sxt/base/error:panic",
    ],
    deps = [
      "//sxt/base/container:span",
    ],
)
