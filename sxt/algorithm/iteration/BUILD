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
        "//sxt/base/iterator:chunk_options",
        "//sxt/base/num:divide_up",
        "//sxt/execution/async:coroutine",
        "//sxt/execution/device:synchronization",
        "//sxt/execution/kernel:kernel_dims",
    ],
)

sxt_cc_component(
    name = "transform",
    test_deps = [
        "//sxt/base/test:unit_test",
        "//sxt/execution/schedule:scheduler",
    ],
    deps = [
        ":for_each",
        "//sxt/algorithm/base:transform_functor",
        "//sxt/base/container:span",
        "//sxt/base/container:span_utility",
        "//sxt/base/device:memory_utility",
        "//sxt/base/device:stream",
        "//sxt/base/error:assert",
        "//sxt/base/iterator:chunk_options",
        "//sxt/base/iterator:index_range",
        "//sxt/base/iterator:index_range_iterator",
        "//sxt/base/iterator:index_range_utility",
        "//sxt/base/macro:cuda_callable",
        "//sxt/base/type:value_type",
        "//sxt/execution/async:coroutine",
        "//sxt/execution/device:device_copy",
        "//sxt/execution/device:for_each",
        "//sxt/execution/device:synchronization",
        "//sxt/memory/resource:async_device_resource",
        "//sxt/memory/resource:chained_resource",
    ],
)
