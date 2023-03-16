load(
    "//bazel:sxt_component.bzl",
    "sxt_cc_component",
)

sxt_cc_component(
    name = "gpu_computation_event",
    test_deps = [
        ":future",
        ":test_kernel",
        "//sxt/execution/base:stream",
        "//sxt/execution/schedule:scheduler",
        "//sxt/base/device:event_utility",
        "//sxt/base/device:memory_utility",
        "//sxt/base/test:unit_test",
        "//sxt/memory/management:managed_array",
        "//sxt/memory/resource:async_device_resource",
        "//sxt/memory/resource:pinned_resource",
    ],
    deps = [
        ":computation_handle",
        ":promise",
        "//sxt/base/device:event",
        "//sxt/execution/schedule:pollable_event",
    ],
)

sxt_cc_component(
    name = "continuation_fn",
    test_deps = [
        "//sxt/base/test:unit_test",
    ],
)

sxt_cc_component(
    name = "continuation_fn_utility",
    with_test = False,
)

sxt_cc_component(
    name = "test_kernel",
    impl_deps = [
        "//sxt/base/device:property",
        "//sxt/base/device:memory_utility",
        "//sxt/base/num:divide_up",
        "//sxt/execution/base:stream",
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
    name = "task",
    with_test = False,
)

sxt_cc_component(
    name = "promise",
    with_test = False,
    deps = [
        ":future_state",
        ":promise_future_base",
        ":task",
        "//sxt/base/error:assert",
    ],
)

sxt_cc_component(
    name = "continuation",
    with_test = False,
    deps = [
        ":continuation_fn",
        ":future_state",
        ":future_state_utility",
        ":promise",
        ":task",
    ],
)

sxt_cc_component(
    name = "leaf_continuation",
    with_test = False,
    deps = [
        ":future_state",
        ":task",
    ],
)

sxt_cc_component(
    name = "computation_handle",
    impl_deps = [
        "//sxt/base/device:synchronization",
        "//sxt/execution/base:stream_handle",
        "//sxt/execution/base:stream_pool",
        "//sxt/execution/base:stream",
        "//sxt/base/error:assert",
    ],
    test_deps = [
        ":test_kernel",
        "//sxt/execution/base:stream_handle",
        "//sxt/execution/base:stream_pool",
        "//sxt/execution/base:stream",
        "//sxt/base/device:memory_utility",
        "//sxt/base/test:unit_test",
        "//sxt/memory/management:managed_array",
        "//sxt/memory/resource:device_resource",
        "//sxt/memory/resource:pinned_resource",
    ],
)

sxt_cc_component(
    name = "promise_future_base",
    impl_deps = [
        "//sxt/base/error:assert",
    ],
    test_deps = [
        "//sxt/base/test:unit_test",
    ],
)

sxt_cc_component(
    name = "future_state",
    with_test = False,
    deps = [
        "//sxt/base/error:assert",
    ],
)

sxt_cc_component(
    name = "future_state_utility",
    with_test = False,
    deps = [
        ":continuation_fn",
        ":future_state",
        "//sxt/base/error:assert",
    ],
)

sxt_cc_component(
    name = "future_fwd",
    with_test = False,
)

sxt_cc_component(
    name = "future",
    test_deps = [
        "//sxt/base/test:unit_test",
    ],
    deps = [
        ":continuation",
        ":continuation_fn",
        ":continuation_fn_utility",
        ":future_fwd",
        ":future_state",
        ":future_state_utility",
        ":leaf_continuation",
        ":promise",
        ":promise_future_base",
    ],
)

sxt_cc_component(
    name = "synchronization",
    with_test = False,
    deps = [
        ":future",
        ":gpu_computation_event",
        ":promise",
        "//sxt/base/device:event",
        "//sxt/base/device:event_utility",
        "//sxt/base/type:raw_stream",
        "//sxt/execution/base:stream",
        "//sxt/execution/schedule:scheduler",
    ],
)
