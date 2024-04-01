load(
    "//bazel:sxt_build_system.bzl",
    "sxt_cc_component",
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
    name = "partition_table_accessor",
    with_test = False,
    deps = [
        "//sxt/base/container:span",
        "//sxt/base/curve:element",
        "//sxt/base/type:raw_stream",
    ],
)

sxt_cc_component(
    name = "in_memory_partition_table_accessor",
    deps = [
      ":partition_table_accessor",
      "//sxt/base/device:memory_utility",
      "//sxt/base/error:assert",
      "//sxt/base/error:panic",
        "//sxt/memory/management:managed_array",
        "//sxt/memory/resource:pinned_resource",
    ],
    test_deps = [
      "//sxt/base/test:unit_test",
    ],
)

sxt_cc_component(
    name = "product_kernel",
    test_deps = [
        "//sxt/base/curve:example_element",
        "//sxt/base/device:stream",
        "//sxt/base/device:synchronization",
        "//sxt/base/test:unit_test",
        "//sxt/memory/resource:managed_device_resource",
    ],
    deps = [
        "//sxt/algorithm/iteration:for_each",
        "//sxt/base/curve:element",
        "//sxt/base/macro:cuda_callable",
        "//sxt/base/num:log2p1",
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
    ],
)

sxt_cc_component(
    name = "partition_index_kernel",
    test_deps = [
        "//sxt/base/device:stream",
        "//sxt/base/device:synchronization",
        "//sxt/base/test:unit_test",
        "//sxt/memory/resource:managed_device_resource",
    ],
    deps = [
        "//sxt/base/type:raw_stream",
    ],
)

sxt_cc_component(
    name = "partition_index",
    impl_deps = [
        ":partition_index_kernel",
        "//sxt/base/container:span_utility",
        "//sxt/base/device:stream",
        "//sxt/base/device:memory_utility",
        "//sxt/base/num:divide_up",
        "//sxt/execution/async:coroutine",
        "//sxt/execution/device:synchronization",
        "//sxt/memory/management:managed_array",
        "//sxt/memory/resource:async_device_resource",
    ],
    test_deps = [
        "//sxt/base/test:unit_test",
        "//sxt/execution/async:future",
        "//sxt/execution/schedule:scheduler",
        "//sxt/memory/resource:managed_device_resource",
    ],
    deps = [
        "//sxt/base/container:span",
        "//sxt/base/error:assert",
        "//sxt/execution/async:future_fwd",
    ],
)

sxt_cc_component(
    name = "multiexponentiation",
    test_deps = [
        "//sxt/base/test:unit_test",
    ],
    deps = [
        ":partition_index",
        ":product_kernel",
        ":reduce",
        "//sxt/base/container:span",
        "//sxt/base/curve:element",
        "//sxt/base/device:memory_utility",
        "//sxt/base/device:stream",
        "//sxt/base/num:divide_up",
        "//sxt/execution/async:coroutine",
        "//sxt/execution/device:synchronization",
        "//sxt/memory/management:managed_array",
        "//sxt/memory/resource:async_device_resource",
        "//sxt/memory/resource:device_resource",
    ],
)
