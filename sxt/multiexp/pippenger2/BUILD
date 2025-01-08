load(
    "//bazel:sxt_build_system.bzl",
    "sxt_cc_component",
)

sxt_cc_component(
    name = "window_width",
    impl_deps = [
        "//sxt/base/error:panic",
        "//sxt/base/log:log",
    ],
    test_deps = [
        "//sxt/base/test:unit_test",
    ],
)

sxt_cc_component(
    name = "combination",
    test_deps = [
        "//sxt/base/curve:example_element",
        "//sxt/base/device:stream",
        "//sxt/base/device:synchronization",
        "//sxt/base/test:unit_test",
        "//sxt/execution/schedule:scheduler",
        "//sxt/memory/resource:managed_device_resource",
    ],
    deps = [
        "//sxt/algorithm/iteration:for_each",
        "//sxt/base/container:span",
        "//sxt/base/container:span_utility",
        "//sxt/base/curve:element",
        "//sxt/base/device:memory_utility",
        "//sxt/base/error:assert",
        "//sxt/base/macro:cuda_callable",
        "//sxt/base/type:raw_stream",
        "//sxt/execution/async:coroutine",
        "//sxt/execution/device:synchronization",
        "//sxt/memory/management:managed_array",
        "//sxt/memory/resource:async_device_resource",
    ],
)

sxt_cc_component(
    name = "combine_reduce",
    test_deps = [
        "//sxt/base/curve:example_element",
        "//sxt/base/device:stream",
        "//sxt/base/device:synchronization",
        "//sxt/base/test:unit_test",
        "//sxt/execution/schedule:scheduler",
        "//sxt/memory/resource:managed_device_resource",
    ],
    deps = [
        "//sxt/algorithm/iteration:for_each",
        "//sxt/base/container:span",
        "//sxt/base/container:span_utility",
        "//sxt/base/curve:element",
        "//sxt/base/error:assert",
        "//sxt/base/iterator:split",
        "//sxt/base/macro:cuda_callable",
        "//sxt/base/type:raw_stream",
        "//sxt/execution/async:coroutine",
        "//sxt/execution/device:copy",
        "//sxt/execution/device:for_each",
        "//sxt/execution/device:synchronization",
        "//sxt/memory/management:managed_array",
        "//sxt/memory/resource:async_device_resource",
    ],
)

sxt_cc_component(
    name = "partition_product",
    test_deps = [
        ":in_memory_partition_table_accessor",
        "//sxt/base/curve:example_element",
        "//sxt/base/device:stream",
        "//sxt/base/device:synchronization",
        "//sxt/base/test:unit_test",
        "//sxt/execution/schedule:scheduler",
        "//sxt/memory/resource:managed_device_resource",
    ],
    deps = [
        "//sxt/algorithm/iteration:for_each",
        "//sxt/base/bit:iteration",
        "//sxt/base/bit:permutation",
        "//sxt/base/container:span",
        "//sxt/base/curve:element",
        "//sxt/base/device:memory_utility",
        "//sxt/base/device:stream",
        "//sxt/base/error:assert",
        "//sxt/base/macro:cuda_callable",
        "//sxt/base/num:divide_up",
        "//sxt/base/num:round_up",
        "//sxt/execution/async:coroutine",
        "//sxt/execution/device:synchronization",
        "//sxt/memory/management:managed_array",
        "//sxt/memory/resource:async_device_resource",
        "//sxt/memory/resource:device_resource",
    ],
)

sxt_cc_component(
    name = "variable_length_partition_product",
    test_deps = [
        ":in_memory_partition_table_accessor",
        "//sxt/base/curve:example_element",
        "//sxt/base/device:stream",
        "//sxt/base/device:synchronization",
        "//sxt/base/test:unit_test",
        "//sxt/execution/schedule:scheduler",
        "//sxt/memory/management:managed_array",
        "//sxt/memory/resource:managed_device_resource",
    ],
    deps = [
        ":partition_product",
        ":partition_table_accessor",
        "//sxt/algorithm/iteration:for_each",
        "//sxt/base/container:span",
        "//sxt/base/curve:element",
        "//sxt/base/device:memory_utility",
        "//sxt/base/num:divide_up",
        "//sxt/base/num:round_up",
        "//sxt/execution/async:coroutine",
        "//sxt/execution/device:synchronization",
        "//sxt/memory/management:managed_array",
        "//sxt/memory/resource:async_device_resource",
        "//sxt/memory/resource:device_resource",
    ],
)

sxt_cc_component(
    name = "partition_table",
    test_deps = [
        "//sxt/base/curve:example_element",
        "//sxt/base/device:stream",
        "//sxt/base/device:synchronization",
        "//sxt/base/test:unit_test",
        "//sxt/memory/resource:managed_device_resource",
    ],
    deps = [
        "//sxt/algorithm/iteration:for_each",
        "//sxt/base/bit:iteration",
        "//sxt/base/bit:permutation",
        "//sxt/base/container:span",
        "//sxt/base/curve:element",
        "//sxt/base/error:assert",
        "//sxt/base/macro:cuda_callable",
    ],
)

sxt_cc_component(
    name = "partition_table_accessor_base",
    with_test = False,
    deps = [
    ],
)

sxt_cc_component(
    name = "partition_table_accessor",
    with_test = False,
    deps = [
        ":partition_table_accessor_base",
        "//sxt/base/container:span",
        "//sxt/base/type:raw_stream",
    ],
)

sxt_cc_component(
    name = "in_memory_partition_table_accessor",
    test_deps = [
        "//sxt/base/curve:example_element",
        "//sxt/base/device:stream",
        "//sxt/base/device:synchronization",
        "//sxt/base/test:temp_file",
        "//sxt/base/test:unit_test",
        "//sxt/memory/resource:device_resource",
    ],
    deps = [
        ":partition_table_accessor",
        "//sxt/base/container:span_utility",
        "//sxt/base/device:memory_utility",
        "//sxt/base/error:assert",
        "//sxt/base/error:panic",
        "//sxt/base/memory:alloc",
        "//sxt/memory/management:managed_array",
        "//sxt/memory/resource:pinned_resource",
    ],
)

sxt_cc_component(
    name = "in_memory_partition_table_accessor_utility",
    test_deps = [
        "//sxt/base/curve:example_element",
        "//sxt/base/device:stream",
        "//sxt/base/device:synchronization",
        "//sxt/base/test:unit_test",
        "//sxt/memory/management:managed_array",
        "//sxt/memory/resource:device_resource",
    ],
    deps = [
        ":in_memory_partition_table_accessor",
        ":partition_table",
        ":window_width",
        "//sxt/base/container:span",
        "//sxt/base/curve:element",
        "//sxt/base/memory:alloc",
        "//sxt/base/num:divide_up",
        "//sxt/memory/resource:pinned_resource",
    ],
)

sxt_cc_component(
    name = "multiexponentiation",
    test_deps = [
        ":in_memory_partition_table_accessor_utility",
        "//sxt/base/curve:example_element",
        "//sxt/base/num:fast_random_number_generator",
        "//sxt/base/test:unit_test",
        "//sxt/curve21/operation:add",
        "//sxt/curve21/operation:double",
        "//sxt/curve21/operation:neg",
        "//sxt/curve21/type:element_p3",
        "//sxt/curve21/type:literal",
        "//sxt/execution/schedule:scheduler",
        "//sxt/ristretto/random:element",
    ],
    deps = [
        ":combination",
        ":combine_reduce",
        ":partition_product",
        ":partition_table_accessor",
        ":reduce",
        ":variable_length_computation",
        "//sxt/base/container:span",
        "//sxt/base/container:span_utility",
        "//sxt/base/curve:element",
        "//sxt/base/device:memory_utility",
        "//sxt/base/device:property",
        "//sxt/base/device:state",
        "//sxt/base/device:stream",
        "//sxt/base/device:synchronization",
        "//sxt/base/iterator:split",
        "//sxt/base/log",
        "//sxt/execution/async:coroutine",
        "//sxt/execution/device:for_each",
        "//sxt/memory/management:managed_array",
        "//sxt/memory/resource:async_device_resource",
        "//sxt/memory/resource:device_resource",
        "//sxt/memory/resource:pinned_resource",
    ],
)

sxt_cc_component(
    name = "variable_length_multiexponentiation",
    test_deps = [
        ":in_memory_partition_table_accessor_utility",
        "//sxt/base/curve:example_element",
        "//sxt/base/test:unit_test",
        "//sxt/execution/schedule:scheduler",
        "//sxt/ristretto/random:element",
    ],
    deps = [
        ":combination",
        ":combine_reduce",
        ":partition_table_accessor",
        ":reduce",
        ":variable_length_computation",
        ":variable_length_partition_product",
        "//sxt/base/container:span",
        "//sxt/base/container:span_utility",
        "//sxt/base/curve:element",
        "//sxt/base/device:memory_utility",
        "//sxt/base/device:property",
        "//sxt/base/device:state",
        "//sxt/base/device:stream",
        "//sxt/base/device:synchronization",
        "//sxt/base/iterator:split",
        "//sxt/base/log",
        "//sxt/execution/async:coroutine",
        "//sxt/execution/device:for_each",
        "//sxt/memory/management:managed_array",
        "//sxt/memory/resource:async_device_resource",
        "//sxt/memory/resource:device_resource",
        "//sxt/memory/resource:pinned_resource",
    ],
)

sxt_cc_component(
    name = "reduce",
    test_deps = [
        "//sxt/base/curve:example_element",
        "//sxt/base/device:stream",
        "//sxt/base/device:synchronization",
        "//sxt/base/test:unit_test",
        "//sxt/memory/resource:managed_device_resource",
    ],
    deps = [
        "//sxt/algorithm/iteration:for_each",
        "//sxt/base/container:span",
        "//sxt/base/curve:element",
        "//sxt/base/device:memory_utility",
        "//sxt/base/macro:cuda_callable",
        "//sxt/base/type:raw_stream",
        "//sxt/memory/management:managed_array",
        "//sxt/memory/resource:async_device_resource",
        "//sxt/memory/resource:pinned_resource",
    ],
)

sxt_cc_component(
    name = "variable_length_computation",
    test_deps = [
        "//sxt/base/test:unit_test",
    ],
    deps = [
        "//sxt/base/container:span",
    ],
)
