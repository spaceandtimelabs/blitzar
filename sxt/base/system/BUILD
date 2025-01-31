load(
    "//bazel:sxt_build_system.bzl",
    "sxt_cc_component",
)

sxt_cc_component(
    name = "file_io",
    test_deps = [
        "//sxt/base/test:temp_file",
        "//sxt/base/test:unit_test",
    ],
    deps = [
        "//sxt/base/container:span",
        "//sxt/base/error:panic",
    ],
)

sxt_cc_component(
    name = "directory_recorder",
    impl_deps = [
        "//sxt/base/error:panic",
    ],
    test_deps = [
        "//sxt/base/test:temp_directory",
        "//sxt/base/test:unit_test",
    ],
)
