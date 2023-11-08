load(
    "//bazel:sxt_build_system.bzl",
    "sxt_cc_component",
)

sxt_cc_component(
    name = "value_cache",
    test_deps = [
        "//sxt/base/test:unit_test",
    ],
    with_test = False,
    deps = [
        "//sxt/base/container:span",
    ],
)

sxt_cc_component(
    name = "value_cache_utility",
    test_deps = [
        "//sxt/base/test:unit_test",
    ],
    with_test = False,
    deps = [
        ":value_cache",
        "//sxt/base/container:span",
        "//sxt/base/error:assert",
    ],
)

sxt_cc_component(
    name = "multiproduct",
    test_deps = [
        ":test_operator",
        ":value_cache_utility",
        "//sxt/base/test:unit_test",
    ],
    deps = [
        ":value_cache",
        "//sxt/base/bit:iteration",
        "//sxt/base/container:span",
        "//sxt/base/error:assert",
    ],
)

sxt_cc_component(
    name = "test_operator",
    with_test = False,
)
