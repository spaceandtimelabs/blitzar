load(
    "//bazel:sxt_component.bzl",
    "sxt_cc_component",
)

sxt_cc_component(
    name = "driver",
    with_test = False,
    deps = [
        "//sxt/base/container:span",
        "//sxt/memory/management:managed_array_fwd",
    ],
)

sxt_cc_component(
    name = "test_driver",
    impl_deps = [
        "//sxt/base/bit:iteration",
        "//sxt/base/bit:span_op",
        "//sxt/base/container:blob_array",
        "//sxt/memory/management:managed_array",
        "//sxt/multiexp/index:index_table",
        "//sxt/base/error:assert",
    ],
    with_test = False,
    deps = [
        ":driver",
    ],
)

sxt_cc_component(
    name = "exponent_aggregates",
    with_test = False,
    deps = [
        "//sxt/base/container:blob_array",
    ],
)

sxt_cc_component(
    name = "radix_log2",
    impl_deps = [
        "//sxt/base/bit:span_op",
        "//sxt/base/num:log2p1",
    ],
    test_deps = [
        "//sxt/base/test:unit_test",
    ],
    deps = [
        "//sxt/base/container:span",
    ],
)

sxt_cc_component(
    name = "exponent_aggregates_computation",
    impl_deps = [
        ":exponent_aggregates",
        "//sxt/base/bit:count",
        "//sxt/base/bit:span_op",
        "//sxt/multiexp/base:exponent_sequence",
    ],
    test_deps = [
        "//sxt/base/test:unit_test",
    ],
    deps = [
        "//sxt/base/container:span",
    ],
)

sxt_cc_component(
    name = "multiproduct_table",
    impl_deps = [
        "//sxt/base/bit:count",
        "//sxt/base/bit:iteration",
        "//sxt/base/bit:span_op",
        "//sxt/base/container:blob_array",
        "//sxt/base/num:divide_up",
        "//sxt/multiexp/base:digit_utility",
        "//sxt/multiexp/base:exponent_sequence",
        "//sxt/multiexp/index:index_table",
        "//sxt/base/error:assert",
    ],
    test_deps = [
        "//sxt/base/container:blob_array",
        "//sxt/multiexp/base:exponent_sequence",
        "//sxt/multiexp/index:index_table",
        "//sxt/base/test:unit_test",
    ],
    deps = [
        "//sxt/base/container:span",
    ],
)

sxt_cc_component(
    name = "multiexponentiation",
    impl_deps = [
        ":driver",
        ":exponent_aggregates",
        ":exponent_aggregates_computation",
        ":multiproduct_table",
        ":radix_log2",
        "//sxt/base/bit:count",
        "//sxt/base/bit:span_op",
        "//sxt/base/container:blob_array",
        "//sxt/base/num:divide_up",
        "//sxt/multiexp/base:digit_utility",
        "//sxt/multiexp/index:index_table",
    ],
    test_deps = [
        ":test_driver",
        "//sxt/base/test:unit_test",
        "//sxt/memory/management:managed_array",
        "//sxt/multiexp/base:exponent_sequence",
        "//sxt/multiexp/random:random_multiexponentiation_descriptor",
        "//sxt/multiexp/random:random_multiexponentiation_generation",
        "//sxt/multiexp/test:compute_uint64_muladd",
        "//sxt/multiexp/test:int_generation",
    ],
    deps = [
        "//sxt/base/container:span",
        "//sxt/memory/management:managed_array_fwd",
    ],
)
