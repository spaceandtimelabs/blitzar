load(
    "//bazel:sxt_build_system.bzl",
    "sxt_cc_component",
)

sxt_cc_component(
    name = "accumulation_kernel",
    test_deps = [
        "//sxt/base/curve:example_element",
        "//sxt/base/device:synchronization",
        "//sxt/base/test:unit_test",
        "//sxt/memory/management:managed_array",
        "//sxt/memory/resource:managed_device_resource",
    ],
    deps = [
        "//sxt/base/curve:element",
        "//sxt/base/num:divide_up",
    ],
)

sxt_cc_component(
    name = "accumulation",
    test_deps = [
        "//sxt/base/curve:example_element",
        "//sxt/base/device:synchronization",
        "//sxt/base/test:unit_test",
        "//sxt/execution/schedule:scheduler",
        "//sxt/memory/resource:managed_device_resource",
    ],
    deps = [
        ":accumulation_kernel",
        ":combination_kernel",
        "//sxt/base/container:span",
        "//sxt/base/curve:element",
        "//sxt/base/device:memory_utility",
        "//sxt/base/device:property",
        "//sxt/base/device:stream",
        "//sxt/base/error:assert",
        "//sxt/base/iterator:index_range",
        "//sxt/base/iterator:index_range_iterator",
        "//sxt/base/iterator:index_range_utility",
        "//sxt/base/num:divide_up",
        "//sxt/execution/async:coroutine",
        "//sxt/execution/async:future",
        "//sxt/execution/device:device_viewable",
        "//sxt/execution/device:for_each",
        "//sxt/memory/management:managed_array",
        "//sxt/memory/resource:async_device_resource",
        "//sxt/memory/resource:pinned_resource",
    ],
)

sxt_cc_component(
    name = "combination_kernel",
    test_deps = [
        "//sxt/base/curve:example_element",
        "//sxt/base/device:synchronization",
        "//sxt/base/test:unit_test",
        "//sxt/memory/management:managed_array",
        "//sxt/memory/resource:managed_device_resource",
    ],
    deps = [
        "//sxt/base/curve:element",
        "//sxt/base/num:divide_up",
        "//sxt/base/num:log2p1",
    ],
)

sxt_cc_component(
    name = "combination",
    test_deps = [
        "//sxt/base/curve:example_element",
        "//sxt/base/test:unit_test",
    ],
    deps = [
        "//sxt/base/container:span",
        "//sxt/base/curve:element",
        "//sxt/base/error:assert",
    ],
)

sxt_cc_component(
    name = "multiexponentiation",
    test_deps = [
        "//sxt/base/curve:example_element",
        "//sxt/base/device:synchronization",
        "//sxt/base/test:unit_test",
        "//sxt/curve21/operation:add",
        "//sxt/curve21/operation:double",
        "//sxt/curve21/operation:neg",
        "//sxt/curve21/type:element_p3",
        "//sxt/curve21/type:literal",
        "//sxt/execution/schedule:scheduler",
        "//sxt/memory/resource:managed_device_resource",
    ],
    deps = [
        ":accumulation",
        ":combination",
        ":combination_kernel",
        "//sxt/base/container:span",
        "//sxt/base/container:span_utility",
        "//sxt/base/curve:element",
        "//sxt/base/device:memory_utility",
        "//sxt/base/device:stream",
        "//sxt/base/iterator:index_range",
        "//sxt/base/iterator:index_range_utility",
        "//sxt/base/num:divide_up",
        "//sxt/execution/async:coroutine",
        "//sxt/execution/async:future",
        "//sxt/execution/device:device_viewable",
        "//sxt/execution/device:synchronization",
        "//sxt/memory/management:managed_array",
        "//sxt/memory/resource:async_device_resource",
        "//sxt/memory/resource:pinned_resource",
        "//sxt/multiexp/base:exponent_sequence",
    ],
)

sxt_cc_component(
    name = "multiproduct_table_kernel",
    test_deps = [
        "//sxt/base/device:synchronization",
        "//sxt/base/test:unit_test",
        "//sxt/memory/resource:managed_device_resource",
    ],
    impl_deps = [
    ],
    deps = [
        "@local_cuda//:cub",
        "//sxt/algorithm/block:runlength_count",
    ],
)

sxt_cc_component(
    name = "multiproduct_table",
    test_deps = [
        "//sxt/base/curve:example_element",
        "//sxt/base/test:unit_test",
        "//sxt/base/device:stream",
        "//sxt/base/device:synchronization",
        "//sxt/execution/async:future",
        "//sxt/execution/schedule:scheduler",
        "//sxt/memory/resource:managed_device_resource",
    ],
    impl_deps = [
      ":multiproduct_table_kernel",
        "//sxt/algorithm/iteration:for_each",
        "//sxt/base/device:memory_utility",
        "//sxt/base/device:stream",
        "//sxt/base/error:assert",
        "//sxt/base/num:divide_up",
        "//sxt/execution/async:coroutine",
        "//sxt/execution/device:synchronization",
        "//sxt/memory/resource:async_device_resource",
        "//sxt/memory/resource:device_resource",
        "//sxt/memory/management:managed_array",
        "//sxt/multiexp/base:scalar_array",
    ],
    deps = [
        "//sxt/memory/management:managed_array_fwd",
        "//sxt/execution/async:future_fwd",
        "//sxt/base/container:span",
    ],
)

sxt_cc_component(
    name = "multiexponentiation2",
    test_deps = [
        "//sxt/base/curve:example_element",
        "//sxt/base/device:synchronization",
        "//sxt/base/test:unit_test",
        "//sxt/curve21/operation:add",
        "//sxt/curve21/operation:double",
        "//sxt/curve21/operation:neg",
        "//sxt/curve21/type:element_p3",
        "//sxt/curve21/type:literal",
        "//sxt/execution/schedule:scheduler",
        "//sxt/memory/resource:managed_device_resource",
    ],
    deps = [
        ":reduce",
        ":sum",
        "//sxt/base/container:span",
        "//sxt/base/container:span_utility",
        "//sxt/base/curve:element",
        "//sxt/base/device:memory_utility",
        "//sxt/base/device:stream",
        "//sxt/base/iterator:index_range",
        "//sxt/base/iterator:index_range_utility",
        "//sxt/base/log:log",
        "//sxt/base/num:divide_up",
        "//sxt/execution/async:coroutine",
        "//sxt/execution/async:future",
        "//sxt/execution/device:device_viewable",
        "//sxt/execution/device:synchronization",
        "//sxt/memory/management:managed_array",
        "//sxt/memory/resource:async_device_resource",
        "//sxt/memory/resource:pinned_resource",
        "//sxt/multiexp/base:exponent_sequence",
    ],
)

sxt_cc_component(
    name = "sum",
    test_deps = [
        "//sxt/base/curve:example_element",
        "//sxt/base/device:synchronization",
        "//sxt/base/test:unit_test",
        "//sxt/execution/schedule:scheduler",
        "//sxt/memory/resource:managed_device_resource",
    ],
    deps = [
      ":multiproduct_table",
      "//sxt/algorithm/iteration:for_each",
        "//sxt/base/container:span",
        "//sxt/base/curve:element",
        "//sxt/base/device:memory_utility",
        "//sxt/base/device:property",
        "//sxt/base/device:stream",
        "//sxt/base/error:assert",
        "//sxt/base/log:log",
        "//sxt/base/iterator:index_range",
        "//sxt/base/iterator:index_range_iterator",
        "//sxt/base/iterator:index_range_utility",
        "//sxt/base/num:divide_up",
        "//sxt/execution/async:coroutine",
        "//sxt/execution/async:future",
        "//sxt/execution/device:device_viewable",
        "//sxt/execution/device:for_each",
        "//sxt/execution/device:synchronization",
        "//sxt/memory/management:managed_array",
        "//sxt/memory/resource:async_device_resource",
        "//sxt/memory/resource:pinned_resource",
    ],
)

sxt_cc_component(
    name = "reduce",
    test_deps = [
        "//sxt/base/curve:example_element",
        "//sxt/base/device:synchronization",
        "//sxt/base/test:unit_test",
        "//sxt/execution/schedule:scheduler",
        "//sxt/memory/management:managed_array",
        "//sxt/memory/resource:managed_device_resource",
    ],
    deps = [
        "//sxt/memory/resource:async_device_resource",
      "//sxt/algorithm/iteration:for_each",
        "//sxt/base/curve:element",
      "//sxt/base/container:span",
      "//sxt/base/device:memory_utility",
      "//sxt/base/device:stream",
      "//sxt/base/macro:cuda_callable",
        "//sxt/execution/async:coroutine",
        "//sxt/execution/device:synchronization",
    ],
)
