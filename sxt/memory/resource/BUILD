load(
    "//bazel:sxt_component.bzl",
    "sxt_cc_component",
)

sxt_cc_component(
    name = "device_resource",
    test_deps = [
        "//sxt/base/test:unit_test",
    ],
    deps = [
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_cuda//cuda:cudart_static",
    ],
)

sxt_cc_component(
    name = "pinned_resource",
    test_deps = [
        "//sxt/base/test:unit_test",
    ],
    with_test = False,
    deps = [
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_cuda//cuda:cudart_static",
    ],
)

sxt_cc_component(
    name = "managed_device_resource",
    test_deps = [
        "//sxt/base/test:unit_test",
    ],
    deps = [
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_cuda//cuda:cudart_static",
    ],
)
