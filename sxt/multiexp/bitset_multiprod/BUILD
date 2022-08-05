load(
    "//bazel:sxt_component.bzl",
    "sxt_cc_component",
)

sxt_cc_component(
    name = "value_cache",
    test_deps = [
        "//sxt/base/test:unit_test",
    ],
    deps = [
        "//sxt/base/container:span",
    ],
)

sxt_cc_component(
    name = "value_cache_utility",
    test_deps = [
        "//sxt/base/test:unit_test",
    ],
    deps = [
        ":value_cache",
        "//sxt/base/container:span",
    ],
)

sxt_cc_component(
    name = "multiproduct",
    test_deps = [
        ":value_cache_utility",
        ":test_operator",
        "//sxt/base/test:unit_test",
    ],
    deps = [
        ":value_cache",
        "//sxt/base/bit:iteration",
        "//sxt/base/container:span",
    ],
)

sxt_cc_component(
    name = "test_operator",
    with_test = False,
)
