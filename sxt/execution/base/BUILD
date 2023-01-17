load(
    "//bazel:sxt_component.bzl",
    "sxt_cc_component",
)

sxt_cc_component(
    name = "stream",
    impl_deps = [
        ":stream_handle",
        ":stream_pool",
    ],
    test_deps = [
        ":stream_handle",
        ":stream_pool",
        "//sxt/base/device:property",
        "//sxt/base/test:unit_test",
    ],
)

sxt_cc_component(
    name = "stream_handle",
    with_test = False,
)

sxt_cc_component(
    name = "stream_pool",
    impl_deps = [
        ":stream_handle",
    ],
    test_deps = [
        ":stream_handle",
        "//sxt/base/device:property",
        "//sxt/base/test:unit_test",
    ],
    deps = [
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_cuda//cuda:cudart_static",
    ],
)
