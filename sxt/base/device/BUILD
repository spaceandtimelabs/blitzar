load(
    "//bazel:sxt_component.bzl",
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
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_cuda//cuda:cudart_static",
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
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_cuda//cuda:cudart_static",
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
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_cuda//cuda:cudart_static",
    ],
)

sxt_cc_component(
    name = "memory_utility",
    impl_deps = [
        ":active_device_guard",
        ":pointer_attributes",
        ":stream",
        ":state",
        "//sxt/base/error:panic",
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_cuda//cuda:cudart_static",
    ],
    test_deps = [
        ":pointer_attributes",
        ":state",
        "//sxt/base/test:unit_test",
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_cuda//cuda:cudart_static",
    ],
    deps = [
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
        ":stream_handle",
        ":stream_pool",
        ":property",
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
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_cuda//cuda:cudart_static",
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
    deps = [
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_cuda//cuda:cudart_static",
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
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_cuda//cuda:cudart_static",
    ],
)
