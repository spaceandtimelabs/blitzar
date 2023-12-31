load(
    "//bazel:sxt_build_system.bzl",
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
        "//sxt/multiexp/base:exponent_sequence",
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
        "//sxt/base/num:abs",
        "//sxt/base/num:constexpr_switch",
        "//sxt/base/num:ceil_log2",
        "//sxt/base/num:power2_equality",
        "//sxt/base/type:int",
        "//sxt/base/bit:count",
        "//sxt/base/bit:span_op",
        "//sxt/multiexp/base:exponent_sequence",
    ],
    test_deps = [
        "//sxt/base/test:unit_test",
    ],
    with_test = False,
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
        "//sxt/base/type:int",
        "//sxt/base/container:blob_array",
        "//sxt/base/container:span_utility",
        "//sxt/base/container:stack_array",
        "//sxt/base/num:abs",
        "//sxt/base/num:divide_up",
        "//sxt/multiexp/base:digit_utility",
        "//sxt/multiexp/base:exponent_sequence",
        "//sxt/multiexp/index:index_table",
        "//sxt/base/error:assert",
        "//sxt/base/num:constexpr_switch",
        "//sxt/base/num:ceil_log2",
        "//sxt/base/num:power2_equality",
    ],
    test_deps = [
        "//sxt/base/container:blob_array",
        "//sxt/base/test:unit_test",
        "//sxt/multiexp/base:exponent_sequence",
        "//sxt/multiexp/base:exponent_sequence_utility",
        "//sxt/multiexp/index:index_table",
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
        "//sxt/base/device:stream",
        "//sxt/base/num:divide_up",
        "//sxt/multiexp/base:exponent_sequence",
        "//sxt/memory/management:managed_array",
        "//sxt/memory/resource:async_device_resource",
        "//sxt/execution/async:future",
        "//sxt/execution/async:coroutine",
    ],
    test_deps = [
        "//sxt/base/device:stream",
        "//sxt/base/device:synchronization",
        "//sxt/base/test:unit_test",
        "//sxt/execution/async:future",
        "//sxt/execution/schedule:scheduler",
        "//sxt/memory/management:managed_array",
        "//sxt/memory/resource:managed_device_resource",
        "//sxt/multiexp/base:exponent_sequence_utility",
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
        "//sxt/base/num:abs",
        "//sxt/base/num:divide_up",
        "//sxt/base/num:constexpr_switch",
        "//sxt/base/num:ceil_log2",
        "//sxt/base/num:power2_equality",
        "//sxt/base/device:memory_utility",
        "//sxt/base/type:int",
        "//sxt/multiexp/base:exponent_sequence",
        "//sxt/memory/management:managed_array",
        "//sxt/memory/resource:async_device_resource",
        "//sxt/execution/async:future",
        "//sxt/execution/device:synchronization",
    ],
    is_cuda = True,
    test_deps = [
        "//sxt/base/device:stream",
        "//sxt/base/device:synchronization",
        "//sxt/base/test:unit_test",
        "//sxt/execution/async:future",
        "//sxt/execution/schedule:scheduler",
        "//sxt/memory/management:managed_array",
        "//sxt/memory/resource:managed_device_resource",
        "//sxt/memory/resource:pinned_resource",
        "//sxt/multiexp/base:exponent_sequence_utility",
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
        "//sxt/multiexp/base:exponent_sequence_utility",
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
