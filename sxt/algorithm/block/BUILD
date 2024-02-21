load(
    "//bazel:sxt_build_system.bzl",
    "sxt_cc_component",
)

sxt_cc_component(
    name = "runlength_count",
    test_deps = [
        "//sxt/base/test:unit_test",
        "//sxt/memory/resource:managed_device_resource",
    ],
    deps = [
        "//sxt/base/device:synchronization",
        "//sxt/base/macro:cuda_callable",
        "@local_cuda//:cub",
    ],
)
