load(
    "//bazel:sxt_build_system.bzl",
    "sxt_cc_component",
)

sxt_cc_component(
    name = "kernel_fit",
    impl_deps = [
        "//sxt/base/error:assert",
        "//sxt/base/num:divide_up",
        "//sxt/execution/kernel:kernel_dims",
    ],
    test_deps = [
        "//sxt/base/num:divide_up",
        "//sxt/base/test:unit_test",
        "//sxt/execution/kernel:kernel_dims",
    ],
)

sxt_cc_component(
    name = "for_each",
    is_cuda = True,
    test_deps = [
        "//sxt/base/device:synchronization",
        "//sxt/base/test:unit_test",
        "//sxt/execution/schedule:scheduler",
        "//sxt/memory/management:managed_array",
        "//sxt/memory/resource:managed_device_resource",
    ],
    deps = [
        ":kernel_fit",
        "//sxt/algorithm/base:index_functor",
        "//sxt/base/device:stream",
        "//sxt/base/num:divide_up",
        "//sxt/execution/async:future",
        "//sxt/execution/device:synchronization",
        "//sxt/execution/kernel:kernel_dims",
    ],
)
