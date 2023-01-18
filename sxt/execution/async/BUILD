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
    ],
    test_deps = [
        ":test_kernel",
        "//sxt/execution/base:stream_handle",
        "//sxt/execution/base:stream_pool",
        "//sxt/execution/base:stream",
        "//sxt/base/device:memory_utility",
        "//sxt/base/test:unit_test",
        "//sxt/base/device:property",
        "//sxt/memory/management:managed_array",
        "//sxt/memory/resource:device_resource",
        "//sxt/memory/resource:pinned_resource",
    ],
)
