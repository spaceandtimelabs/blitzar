load(
    "//bazel:sxt_build_system.bzl",
    "sxt_cc_component",
)

sxt_cc_component(
    name = "allocator_aware",
    impl_deps = [
        "//sxt/base/error:assert",
    ],
    with_test = False,
)

sxt_cc_component(
    name = "temp_directory",
    impl_deps = [
    ],
    with_test = False,
)

sxt_cc_component(
    name = "temp_file",
    impl_deps = [
    ],
    with_test = False,
)

sxt_cc_component(
    name = "unit_test",
    with_test = False,
    deps = [
        "@com_github_catchorg_catch2//:catch2",
        "@com_github_catchorg_catch2//:catch2_main",
    ],
)
