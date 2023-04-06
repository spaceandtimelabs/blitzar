load(
    "//bazel:sxt_component.bzl",
    "sxt_cc_component",
)

sxt_cc_component(
    name = "thread_reduction",
    is_cuda = True,
    test_deps = [
        "//sxt/base/test:unit_test",
    ],
    deps = [
        ":warp_reduction",
        "//sxt/algorithm/base:mapper",
        "//sxt/algorithm/base:reducer",
        "//sxt/algorithm/base:reducer_utility",
    ],
)

sxt_cc_component(
    name = "warp_reduction",
    is_cuda = True,
    with_test = False,
    deps = [
        "//sxt/algorithm/base:reducer",
    ],
)

sxt_cc_component(
    name = "reduction",
    is_cuda = True,
    test_deps = [
        ":test_reducer",
        "//sxt/base/test:unit_test",
        "//sxt/base/device:synchronization",
        "//sxt/execution/schedule:scheduler",
        "//sxt/memory/resource:managed_device_resource",
    ],
    deps = [
        ":kernel_fit",
        ":thread_reduction",
        "//sxt/algorithm/base:identity_mapper",
        "//sxt/algorithm/base:mapper",
        "//sxt/algorithm/base:reducer",
        "//sxt/base/device:memory_utility",
        "//sxt/base/device:stream",
        "//sxt/execution/async:future",
        "//sxt/execution/async:synchronization",
        "//sxt/execution/kernel:kernel_dims",
        "//sxt/execution/kernel:launch",
        "//sxt/memory/management:managed_array",
        "//sxt/memory/resource:async_device_resource",
        "//sxt/memory/resource:pinned_resource",
    ],
)

sxt_cc_component(
    name = "test_reducer",
    is_cuda = True,
    with_test = False,
    deps = [
        "//sxt/base/macro:cuda_callable",
    ],
)

sxt_cc_component(
    name = "kernel_fit",
    impl_deps = [
        "//sxt/execution/kernel:kernel_dims",
        "//sxt/base/error:assert",
    ],
    test_deps = [
        "//sxt/execution/kernel:kernel_dims",
        "//sxt/base/test:unit_test",
    ],
)
