load(
    "//bazel:sxt_component.bzl",
    "sxt_cc_component",
)

sxt_cc_component(
    name = "exponent",
    impl_deps = [
        "//sxt/base/bit:count",
    ],
    test_deps = [
        "//sxt/base/test:unit_test",
    ],
)

sxt_cc_component(
    name = "digit_utility",
    deps = [
        "//sxt/base/container:span",
    ],
    impl_deps = [
        "//sxt/base/bit:count",
        "//sxt/base/num:divide_up",
    ],
    test_deps = [
        "//sxt/base/test:unit_test",
    ],
)

sxt_cc_component(
    name = "exponent_utility",
    deps = [
        "//sxt/base/container:span",
    ],
    impl_deps = [
        ":digit_utility",
        ":exponent",
    ],
    test_deps = [
        ":exponent",
        "//sxt/base/test:unit_test",
    ],
)

sxt_cc_component(
    name = "exponent_sequence",
    with_test = False,
)
