load(
    "//bazel:sxt_build_system.bzl",
    "sxt_cc_component",
)

sxt_cc_component(
    name = "active_device_guard",
    impl_deps = [
        ":state",
    ],
    test_deps = [
        ":property",
        "//sxt/base/test:unit_test",
    ],
)

sxt_cc_component(
    name = "pinned_memory_handle",
    with_test = False,
    deps = [
    ],
)

sxt_cc_component(
    name = "pinned_memory_pool",
    deps = [
    ],
    impl_deps = [
      ":pinned_memory_handle",
    ],
)

sxt_cc_component(
    name = "pointer_attributes",
    with_test = False,
    deps = [
        ":pointer_kind",
    ],
)

sxt_cc_component(
    name = "pointer_kind",
    with_test = False,
)

sxt_cc_component(
    name = "event",
    impl_deps = [
        "//sxt/base/error:panic",
    ],
    test_deps = [
        "//sxt/base/test:unit_test",
    ],
    deps = [
        "//sxt/base/type:raw_cuda_event",
    ],
)

sxt_cc_component(
    name = "event_utility",
    impl_deps = [
        "//sxt/base/error:panic",
    ],
    with_test = False,
    deps = [
        "//sxt/base/type:raw_cuda_event",
        "//sxt/base/type:raw_stream",
    ],
)

sxt_cc_component(
    name = "device_map",
    test_deps = [
        "//sxt/base/test:unit_test",
    ],
    deps = [
        ":property",
        "//sxt/base/error:assert",
        "//sxt/base/macro:max_devices",
    ],
)

sxt_cc_component(
    name = "property",
    impl_deps = [
        "//sxt/base/error:panic",
    ],
    test_deps = [
        "//sxt/base/test:unit_test",
    ],
    deps = [
        "//sxt/base/type:raw_stream",
    ],
)

sxt_cc_component(
    name = "memory_utility",
    impl_deps = [
        ":active_device_guard",
        ":stream",
        ":state",
        "//sxt/base/error:panic",
    ],
    test_deps = [
        ":pointer_attributes",
        ":state",
        ":stream",
        "//sxt/base/test:unit_test",
    ],
    deps = [
        ":pointer_attributes",
        "//sxt/base/concept:memcpyable_ranges",
        "//sxt/base/container:span",
        "//sxt/base/error:assert",
        "//sxt/base/type:raw_stream",
    ],
)

sxt_cc_component(
    name = "stream",
    impl_deps = [
        ":stream_handle",
        ":stream_pool",
        "//sxt/base/error:assert",
    ],
    test_deps = [
        ":property",
        ":stream_handle",
        ":stream_pool",
        "//sxt/base/test:unit_test",
    ],
    deps = [
        ":state",
        "//sxt/base/type:raw_stream",
    ],
)

sxt_cc_component(
    name = "stream_handle",
    with_test = False,
    deps = [
        "//sxt/base/type:raw_stream",
    ],
)

sxt_cc_component(
    name = "stream_pool",
    impl_deps = [
        ":active_device_guard",
        ":stream_handle",
        "//sxt/base/error:assert",
        "//sxt/base/error:panic",
    ],
    test_deps = [
        ":stream_handle",
        "//sxt/base/device:property",
        "//sxt/base/test:unit_test",
    ],
    deps = [
        ":device_map",
    ],
)

sxt_cc_component(
    name = "state",
    impl_deps = [
        "//sxt/base/error:panic",
    ],
    test_deps = [
        ":property",
        "//sxt/base/test:unit_test",
    ],
)

sxt_cc_component(
    name = "synchronization",
    impl_deps = [
        "//sxt/base/error:panic",
    ],
    with_test = False,
    deps = [
        "//sxt/base/type:raw_stream",
    ],
)
