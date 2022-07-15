load(
    "//bazel:sxt_component.bzl",
    "sxt_cc_component",
)

sxt_cc_component(
    name = "counting_iterator",
    test_deps = [
        "//sxt/base/test:unit_test",
    ],
    deps = [
        ":iterator_facade",
    ],
)

sxt_cc_component(
    name = "iterator_facade",
    test_deps = [
        "//sxt/base/test:unit_test",
    ],
    deps = [
        "//sxt/base/type:remove_cvref",
    ],
)
