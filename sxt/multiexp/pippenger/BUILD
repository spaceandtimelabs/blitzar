load(
    "//bazel:sxt_component.bzl",
    "sxt_cc_component",
)

sxt_cc_component(
    name = "driver",
    with_test = False,
    deps = [
        "//sxt/base/container:span",
        "//sxt/execution/async:future_fwd",
        "//sxt/memory/management:managed_array_fwd",
    ],
)

sxt_cc_component(
    name = "test_driver",
    impl_deps = [
        "//sxt/base/bit:span_op",
        "//sxt/base/container:span",
        "//sxt/base/container:span_void",
        "//sxt/base/error:assert",
        "//sxt/base/container:blob_array",
        "//sxt/execution/async:future",
        "//sxt/memory/management:managed_array",
        "//sxt/multiexp/index:index_table",
        "//sxt/multiexp/base:generator_utility",
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
    name = "multiproduct_decomposition_gpu",
    impl_deps = [
        ":multiproduct_decomposition_kernel",
        "//sxt/base/error:assert",
        "//sxt/base/device:memory_utility",
        "//sxt/base/num:divide_up",
        "//sxt/multiexp/base:exponent_sequence",
        "//sxt/memory/management:managed_array",
        "//sxt/memory/resource:async_device_resource",
        "//sxt/execution/async:future",
        "//sxt/execution/async:coroutine",
        "//sxt/execution/async:synchronization",
    ],
    test_deps = [
        "//sxt/memory/management:managed_array",
        "//sxt/memory/resource:managed_device_resource",
        "//sxt/multiexp/base:exponent_sequence_utility",
        "//sxt/base/test:unit_test",
        "//sxt/base/device:synchronization",
        "//sxt/execution/schedule:scheduler",
        "//sxt/execution/async:future",
        "//sxt/base/device:stream",
    ],
    deps = [
        "//sxt/base/container:span",
        "//sxt/base/type:raw_stream",
        "//sxt/execution/async:future_fwd",
        "//sxt/memory/management:managed_array_fwd",
    ],
)

sxt_cc_component(
    name = "multiproduct_decomposition_kernel",
    impl_deps = [
        "//sxt/base/error:assert",
        "//sxt/base/num:divide_up",
        "//sxt/base/device:memory_utility",
        "//sxt/multiexp/base:exponent_sequence",
        "//sxt/memory/management:managed_array",
        "//sxt/memory/resource:async_device_resource",
        "//sxt/execution/async:future",
        "//sxt/execution/async:synchronization",
    ],
    is_cuda = True,
    test_deps = [
        "//sxt/base/device:synchronization",
        "//sxt/execution/schedule:scheduler",
        "//sxt/execution/async:future",
        "//sxt/base/device:stream",
        "//sxt/memory/management:managed_array",
        "//sxt/memory/resource:pinned_resource",
        "//sxt/memory/resource:managed_device_resource",
        "//sxt/multiexp/base:exponent_sequence_utility",
        "//sxt/base/test:unit_test",
    ],
    deps = [
        "//sxt/base/container:span",
        "//sxt/execution/async:future_fwd",
        "//sxt/memory/management:managed_array_fwd",
    ],
)

sxt_cc_component(
    name = "multiexponentiation",
    impl_deps = [
        ":driver",
        ":exponent_aggregates",
        ":exponent_aggregates_computation",
        ":multiproduct_table",
        "//sxt/base/bit:span_op",
        "//sxt/base/container:blob_array",
        "//sxt/base/num:divide_up",
        "//sxt/memory/management:managed_array",
        "//sxt/execution/async:future",
        "//sxt/multiexp/base:digit_utility",
        "//sxt/multiexp/index:index_table",
    ],
    test_deps = [
        ":test_driver",
        "//sxt/base/test:unit_test",
        "//sxt/execution/async:future",
        "//sxt/memory/management:managed_array",
        "//sxt/multiexp/base:exponent_sequence",
        "//sxt/multiexp/random:random_multiexponentiation_descriptor",
        "//sxt/multiexp/random:random_multiexponentiation_generation",
        "//sxt/multiexp/test:compute_uint64_muladd",
    ],
    deps = [
        "//sxt/base/container:span",
        "//sxt/base/container:span_void",
        "//sxt/execution/async:future_fwd",
        "//sxt/memory/management:managed_array_fwd",
    ],
)
