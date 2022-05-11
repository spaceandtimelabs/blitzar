load(
    "//bazel:sxt_component.bzl",
    "sxt_cc_component",
)

sxt_cc_component(
    name = "device_resource",
    deps = [
        "@local_config_cuda//cuda:cuda",
    ],
    test_deps = [
        "//sxt/base/test:unit_test",
    ],
)
