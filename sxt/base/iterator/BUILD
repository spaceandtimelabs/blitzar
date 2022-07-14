load(
    "//bazel:sxt_component.bzl",
    "sxt_cc_component",
)

sxt_cc_component(
    name = "counting_iterator",
    deps = [
        ":iterator_facade",
    ],
    test_deps = [
        "//sxt/base/test:unit_test",
    ],
)

sxt_cc_component(
    name = "iterator_facade",
    deps = [
        "//sxt/base/type:remove_cvref",
    ],
    test_deps = [
        "//sxt/base/test:unit_test",
    ],
)
