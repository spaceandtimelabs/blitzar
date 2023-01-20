load(
    "//bazel:sxt_component.bzl",
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
    name = "unit_test",
    with_test = False,
    deps = [
        "@com_github_catchorg_catch2//:catch2",
        "@com_github_catchorg_catch2//:catch2_main",
    ],
)
