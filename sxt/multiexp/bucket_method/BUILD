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
    name = "bucket_descriptor",
    with_test = False,
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
    name = "count",
    test_deps = [
        "//sxt/memory/management:managed_array",
        "//sxt/memory/resource:managed_device_resource",
        "//sxt/base/device:stream",
        "//sxt/base/device:synchronization",
        "//sxt/base/test:unit_test",
    ],
    impl_deps = [
        "//sxt/memory/resource:async_device_resource",
        "//sxt/memory/resource:chained_resource",
        "//sxt/memory/resource:pinned_resource",
        "//sxt/memory/management:managed_array",
        "//sxt/multiexp/base:digit_utility",
        "//sxt/base/device:memory_utility",
        "//sxt/base/device:stream",
        "//sxt/base/error:assert",
        "//sxt/base/num:divide_up",
    ],
    deps = [
        "//sxt/base/container:span",
        "//sxt/memory/management:managed_array_fwd",
    ],
)

sxt_cc_component(
    name = "multiproduct_table",
    test_deps = [
      ":bucket_descriptor",
        "//sxt/execution/async:future",
        "//sxt/execution/schedule:scheduler",
        "//sxt/memory/management:managed_array",
        "//sxt/memory/resource:managed_device_resource",
        "//sxt/base/device:synchronization",
        "//sxt/base/device:stream",
        "//sxt/base/test:unit_test",
    ],
    impl_deps = [
      ":bucket_descriptor",
      ":count",
      "//sxt/multiexp/base:digit_utility",
        "//sxt/algorithm/iteration:for_each",
        "//sxt/algorithm/transform:prefix_sum",
        "//sxt/execution/async:coroutine",
        "//sxt/execution/device:synchronization",
        "//sxt/memory/management:managed_array",
        "//sxt/memory/resource:async_device_resource",
        "//sxt/multiexp/base:scalar_array",
        "//sxt/base/container:span_utility",
        "//sxt/base/device:memory_utility",
        "//sxt/base/device:stream",
        "//sxt/base/error:assert",
        "//sxt/base/num:divide_up",
        "@local_cuda//:cub",
    ],
    deps = [
        "//sxt/execution/async:future_fwd",
        "//sxt/base/container:span",
        "//sxt/memory/management:managed_array_fwd",
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
      ":sum",
      ":sum2",
      ":reduction",
        "//sxt/base/container:span",
        "//sxt/base/container:span_utility",
        "//sxt/base/curve:element",
        "//sxt/base/device:memory_utility",
        "//sxt/base/device:stream",
        "//sxt/base/iterator:index_range",
        "//sxt/base/iterator:index_range_iterator",
        "//sxt/base/iterator:index_range_utility",
        "//sxt/base/num:divide_up",
        "//sxt/execution/async:coroutine",
        "//sxt/execution/async:future",
        "//sxt/execution/device:for_each",
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
      "//sxt/base/error:assert",
        "//sxt/base/test:unit_test",
        "//sxt/execution/schedule:scheduler",
        "//sxt/memory/resource:managed_device_resource",
    ],
    impl_deps = [
      "//sxt/algorithm/iteration:kernel_fit",
    ],
    deps = [
      ":bucket_descriptor",
      ":multiproduct_table",
      "//sxt/algorithm/iteration:for_each",
        "//sxt/base/container:span",
        "//sxt/base/device:memory_utility",
        "//sxt/base/device:stream",
        "//sxt/execution/async:coroutine",
        "//sxt/execution/device:device_viewable",
        "//sxt/memory/management:managed_array",
        "//sxt/memory/resource:async_device_resource",
        "//sxt/memory/resource:device_resource",
        "//sxt/memory/resource:pinned_resource",
    ],
)

sxt_cc_component(
    name = "reduction",
    test_deps = [
      "//sxt/algorithm/iteration:for_each",
      "//sxt/base/curve:example_element",
      "//sxt/base/device:memory_utility",
      "//sxt/base/device:synchronization",
      "//sxt/base/error:assert",
        "//sxt/base/test:unit_test",
        "//sxt/memory/resource:managed_device_resource",
    ],
    impl_deps = [
    ],
    deps = [
        "//sxt/memory/management:managed_array",
        "//sxt/memory/resource:async_device_resource",
      "//sxt/base/device:stream",
      "//sxt/base/error:assert",
      "//sxt/base/macro:cuda_callable",
      "//sxt/algorithm/iteration:for_each",
        "//sxt/base/container:span",
    ],
)

sxt_cc_component(
    name = "sum2",
    test_deps = [
      "//sxt/base/curve:example_element",
      "//sxt/base/device:synchronization",
      "//sxt/base/error:assert",
        "//sxt/base/test:unit_test",
        "//sxt/memory/resource:managed_device_resource",
    ],
    deps = [
      "//sxt/algorithm/iteration:for_each",
        "//sxt/multiexp/base:scalar_array",
        "//sxt/base/container:span",
        "//sxt/base/device:memory_utility",
        "//sxt/base/device:stream",
        "//sxt/base/num:divide_up",
        "//sxt/execution/async:coroutine",
        "//sxt/execution/device:device_viewable",
        "//sxt/memory/management:managed_array",
        "//sxt/memory/resource:async_device_resource",
        "//sxt/memory/resource:device_resource",
        "//sxt/memory/resource:pinned_resource",
        "//sxt/multiexp/base:digit_utility",
    ],
)
