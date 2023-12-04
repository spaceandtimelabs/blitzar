load(
    "//bazel:sxt_build_system.bzl",
    "sxt_cc_component",
)

sxt_cc_component(
    name = "blob_array",
    test_deps = [
        "//sxt/base/test:unit_test",
    ],
    deps = [
        ":span",
        ":span_iterator",
    ],
)

sxt_cc_component(
    name = "span",
    test_deps = [
        "//sxt/base/test:unit_test",
    ],
    deps = [
        ":span_void",
        "//sxt/base/error:assert",
        "//sxt/base/macro:cuda_callable",
        "//sxt/base/macro:cuda_warning",
    ],
)

sxt_cc_component(
    name = "span_utility",
    deps = [
        ":span",
    ],
    test_deps = [
        "//sxt/base/test:unit_test",
    ],
)

sxt_cc_component(
    name = "span_void",
    test_deps = [
        "//sxt/base/test:unit_test",
    ],
    deps = [
        "//sxt/base/error:assert",
    ],
)

sxt_cc_component(
    name = "span_iterator",
    test_deps = [
        "//sxt/base/test:unit_test",
    ],
    deps = [
        ":span",
        "//sxt/base/iterator:iterator_facade",
    ],
)

sxt_cc_component(
    name = "stack_array",
    test_deps = [
        "//sxt/base/test:unit_test",
    ],
    deps = [
        ":span",
    ],
)
