load(
    "//bazel:sxt_build_system.bzl",
    "sxt_cc_component",
)

sxt_cc_component(
    name = "alloc",
    with_test = False,
)

sxt_cc_component(
    name = "alloc_utility",
    test_deps = [
        "//sxt/base/test:unit_test",
    ],
    deps = [
        ":alloc",
    ],
)
