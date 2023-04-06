load(
    "//bazel:sxt_component.bzl",
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
        "//sxt/execution/kernel:kernel_dims",
        "//sxt/base/test:unit_test",
    ],
)

sxt_cc_component(
    name = "for_each",
    is_cuda = True,
    test_deps = [
        "//sxt/base/test:unit_test",
        "//sxt/base/device:synchronization",
        "//sxt/memory/management:managed_array",
        "//sxt/memory/resource:managed_device_resource",
        "//sxt/execution/schedule:scheduler",
    ],
    deps = [
        ":kernel_fit",
        "//sxt/algorithm/base:index_functor",
        "//sxt/base/device:stream",
        "//sxt/base/num:divide_up",
        "//sxt/execution/async:future",
        "//sxt/execution/async:synchronization",
        "//sxt/execution/kernel:kernel_dims",
    ],
)
