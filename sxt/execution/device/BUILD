load(
    "//bazel:sxt_build_system.bzl",
    "sxt_cc_component",
)

sxt_cc_component(
    name = "device_viewable",
    test_deps = [
        "//sxt/base/device:active_device_guard",
        "//sxt/base/device:property",
        "//sxt/base/test:unit_test",
        "//sxt/execution/schedule:scheduler",
        "//sxt/memory/resource:chained_resource",
        "//sxt/memory/resource:device_resource",
    ],
    deps = [
        ":event_future",
        ":synchronization",
        "//sxt/base/container:span",
        "//sxt/base/device:event",
        "//sxt/base/device:event_utility",
        "//sxt/base/device:memory_utility",
        "//sxt/base/device:pointer_attributes",
        "//sxt/base/device:state",
        "//sxt/base/device:stream",
        "//sxt/base/type:value_type",
        "//sxt/execution/async:future",
        "//sxt/memory/management:managed_array",
    ],
)

sxt_cc_component(
    name = "device_copy",
    with_test = False,
    # test_deps = [
    #     "//sxt/base/device:active_device_guard",
    #     "//sxt/base/device:property",
    #     "//sxt/base/test:unit_test",
    #     "//sxt/execution/schedule:scheduler",
    #     "//sxt/memory/resource:device_resource",
    # ],
    deps = [
        ":event_future",
        ":synchronization",
        "//sxt/base/container:span",
        "//sxt/base/container:span_utility",
        "//sxt/base/device:event",
        "//sxt/base/device:event_utility",
        "//sxt/base/device:memory_utility",
        "//sxt/base/device:pointer_attributes",
        "//sxt/base/device:state",
        "//sxt/base/device:stream",
        "//sxt/base/type:value_type",
        "//sxt/execution/async:future",
        "//sxt/memory/management:managed_array",
    ],
)

sxt_cc_component(
    name = "event_future",
    test_deps = [
        "//sxt/base/device:event_utility",
        "//sxt/base/device:stream",
        "//sxt/base/test:unit_test",
        "//sxt/execution/async:coroutine",
        "//sxt/memory/management:managed_array",
    ],
    deps = [
        ":computation_event",
        ":computation_handle",
        "//sxt/base/device:event",
        "//sxt/execution/async:future",
        "//sxt/execution/schedule:scheduler",
    ],
)

sxt_cc_component(
    name = "for_each",
    impl_deps = [
        ":available_device",
        "//sxt/execution/async:future",
        "//sxt/execution/async:coroutine",
        "//sxt/base/device:active_device_guard",
        "//sxt/base/device:property",
        "//sxt/base/iterator:index_range",
        "//sxt/base/iterator:index_range_iterator",
        "//sxt/base/iterator:index_range_utility",
    ],
    test_deps = [
        "//sxt/base/iterator:index_range",
        "//sxt/base/test:unit_test",
        "//sxt/execution/async:future",
    ],
    deps = [
        "//sxt/execution/async:future_fwd",
        "//sxt/execution/schedule:scheduler",
    ],
)

sxt_cc_component(
    name = "available_device",
    impl_deps = [
        "//sxt/execution/async:future",
        "//sxt/execution/schedule:scheduler",
        "//sxt/execution/schedule:pending_event",
    ],
    test_deps = [
        ":synchronization",
        ":test_kernel",
        "//sxt/base/device:active_device_guard",
        "//sxt/base/device:stream",
        "//sxt/base/test:unit_test",
        "//sxt/execution/async:coroutine",
        "//sxt/memory/management:managed_array",
    ],
    deps = [
        "//sxt/execution/async:future_fwd",
    ],
)

sxt_cc_component(
    name = "test_kernel",
    impl_deps = [
        "//sxt/base/device:property",
        "//sxt/base/device:memory_utility",
        "//sxt/base/num:divide_up",
        "//sxt/base/device:stream",
        "//sxt/memory/resource:async_device_resource",
        "//sxt/memory/management:managed_array",
    ],
    is_cuda = True,
    with_test = False,
    deps = [
        "//sxt/base/type:raw_stream",
    ],
)

sxt_cc_component(
    name = "computation_event",
    test_deps = [
        ":test_kernel",
        "//sxt/base/device:event_utility",
        "//sxt/base/device:memory_utility",
        "//sxt/base/device:stream",
        "//sxt/base/test:unit_test",
        "//sxt/execution/async:future",
        "//sxt/execution/schedule:scheduler",
        "//sxt/memory/management:managed_array",
        "//sxt/memory/resource:async_device_resource",
        "//sxt/memory/resource:pinned_resource",
    ],
    deps = [
        ":computation_handle",
        "//sxt/base/device:event",
        "//sxt/execution/async:promise",
        "//sxt/execution/schedule:pollable_event",
    ],
)

sxt_cc_component(
    name = "computation_handle",
    impl_deps = [
        "//sxt/base/device:synchronization",
        "//sxt/base/device:stream_handle",
        "//sxt/base/device:stream_pool",
        "//sxt/base/device:stream",
        "//sxt/base/error:assert",
    ],
    test_deps = [
        ":test_kernel",
        "//sxt/base/device:memory_utility",
        "//sxt/base/device:stream",
        "//sxt/base/device:stream_handle",
        "//sxt/base/device:stream_pool",
        "//sxt/base/test:unit_test",
        "//sxt/memory/management:managed_array",
        "//sxt/memory/resource:device_resource",
        "//sxt/memory/resource:pinned_resource",
    ],
)

sxt_cc_component(
    name = "synchronization",
    test_deps = [
        ":test_kernel",
        "//sxt/base/test:unit_test",
        "//sxt/execution/async:coroutine",
        "//sxt/memory/management:managed_array",
    ],
    deps = [
        ":computation_event",
        ":event_future",
        "//sxt/base/device:event",
        "//sxt/base/device:event_utility",
        "//sxt/base/device:property",
        "//sxt/base/device:stream",
        "//sxt/base/type:raw_stream",
        "//sxt/execution/async:future",
        "//sxt/execution/async:promise",
        "//sxt/execution/schedule:scheduler",
    ],
)
