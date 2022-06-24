load(
    "//bazel:sxt_component.bzl",
    "sxt_cc_component",
)

sxt_cc_component(
    name = "driver",
    with_test = False,
    deps = [
        "//sxt/memory/management:managed_array_fwd",
        "//sxt/base/container:span",
    ],
)

sxt_cc_component(
    name = "test_driver",
    with_test = False,
    deps = [
        ":driver",
    ],
    impl_deps = [
        "//sxt/base/bit:iteration",
        "//sxt/memory/management:managed_array",
        "//sxt/multiexp/index:index_table",
    ],
)

sxt_cc_component(
    name = "exponent_aggregates",
    with_test = False,
    deps = [
        "//sxt/multiexp/base:exponent",
    ],
)

sxt_cc_component(
    name = "radix_log2",
    impl_deps = [
        "//sxt/multiexp/base:exponent",
        "//sxt/base/num:log2p1",
        "//sxt/base/container:span"
    ],
    test_deps = [
        "//sxt/base/test:unit_test",
        "//sxt/multiexp/base:exponent"
    ],
)

sxt_cc_component(
    name = "exponent_aggregates_computation",
    deps = [
        "//sxt/base/container:span",
    ],
    impl_deps = [
        ":exponent_aggregates",
        "//sxt/base/bit:count",
        "//sxt/multiexp/base:exponent",
        "//sxt/multiexp/base:exponent_utility",
        "//sxt/multiexp/base:exponent_sequence",
    ],
    test_deps = [
        "//sxt/base/test:unit_test",
    ],
)

sxt_cc_component(
    name = "multiproduct_table",
    deps = [
        "//sxt/base/container:span",
    ],
    impl_deps = [
        "//sxt/base/bit:count",
        "//sxt/base/bit:iteration",
        "//sxt/multiexp/base:digit_utility",
        "//sxt/multiexp/base:exponent",
        "//sxt/multiexp/base:exponent_utility",
        "//sxt/multiexp/base:exponent_sequence",
        "//sxt/multiexp/index:index_table",
    ],
    test_deps = [
        "//sxt/multiexp/base:exponent",
        "//sxt/multiexp/base:exponent_sequence",
        "//sxt/multiexp/index:index_table",
        "//sxt/base/test:unit_test",
    ],
)

sxt_cc_component(
    name = "multiexponentiation",
    deps = [
        "//sxt/base/container:span",
        "//sxt/memory/management:managed_array_fwd",
    ],
    impl_deps = [
        ":driver",
        ":exponent_aggregates",
        ":exponent_aggregates_computation",
        ":multiproduct_table",
        ":radix_log2",
        "//sxt/base/bit:count",
        "//sxt/multiexp/base:exponent_utility",
        "//sxt/multiexp/index:index_table",
    ],
    test_deps = [
        ":test_driver",
        "//sxt/base/test:unit_test",
        "//sxt/memory/management:managed_array",
        "//sxt/multiexp/base:exponent_sequence",
    ],
)
