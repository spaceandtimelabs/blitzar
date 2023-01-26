load(
    "//bazel:sxt_component.bzl",
    "sxt_cc_component",
)

sxt_cc_component(
    name = "test_kernel",
    impl_deps = [
        "//sxt/base/num:divide_up",
        "//sxt/execution/base:stream",
    ],
    is_cuda = True,
    with_test = False,
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
    name = "future_value_storage",
    test_deps = [
        "//sxt/base/test:unit_test",
    ],
)

sxt_cc_component(
    name = "future_completion_fn",
    with_test = False,
)

sxt_cc_component(
    name = "future_ready",
    with_test = False,
)

sxt_cc_component(
    name = "future_fwd",
    with_test = False,
)

sxt_cc_component(
    name = "future",
    test_deps = [
        ":test_kernel",
        "//sxt/execution/base:stream",
        "//sxt/base/device:memory_utility",
        "//sxt/base/test:unit_test",
        "//sxt/memory/management:managed_array",
        "//sxt/memory/resource:device_resource",
        "//sxt/memory/resource:pinned_resource",
    ],
    deps = [
        ":computation_handle",
        ":future_completion_fn",
        ":future_fwd",
        ":future_ready",
        ":future_value_storage",
    ],
)
