load(
    "//bazel:sxt_build_system.bzl",
    "sxt_cc_component",
)

sxt_cc_component(
    name = "block_computation_descriptor",
    with_test = False,
    deps = [
        "//sxt/execution/kernel:block_size",
    ],
)

sxt_cc_component(
    name = "completion",
    test_deps = [
        "//sxt/algorithm/reduction:test_reducer",
        "//sxt/base/test:unit_test",
    ],
    deps = [
        ":block_computation_descriptor",
        "//sxt/algorithm/base:reducer",
        "//sxt/base/container:span",
        "//sxt/base/error:assert",
    ],
)

sxt_cc_component(
    name = "multiproduct_computation_descriptor",
    with_test = False,
    deps = [
        ":block_computation_descriptor",
        "//sxt/execution/kernel:block_size",
        "//sxt/memory/management:managed_array",
    ],
)

sxt_cc_component(
    name = "computation_setup",
    impl_deps = [
        ":multiproduct_computation_descriptor",
        "//sxt/algorithm/reduction:kernel_fit",
        "//sxt/execution/kernel:kernel_dims",
    ],
    test_deps = [
        ":multiproduct_computation_descriptor",
        "//sxt/memory/management:managed_array",
        "//sxt/base/test:unit_test",
    ],
    deps = [
        "//sxt/base/container:span",
    ],
)

sxt_cc_component(
    name = "kernel",
    is_cuda = True,
    test_deps = [
        "//sxt/algorithm/base:gather_mapper",
        "//sxt/algorithm/reduction:test_reducer",
        "//sxt/memory/management:managed_array",
        "//sxt/memory/resource:managed_device_resource",
        "//sxt/base/device:synchronization",
        "//sxt/base/test:unit_test",
    ],
    deps = [
        ":block_computation_descriptor",
        "//sxt/algorithm/base:reducer",
        "//sxt/algorithm/reduction:thread_reduction",
        "//sxt/execution/kernel:launch",
    ],
)

sxt_cc_component(
    name = "multiproduct",
    is_cuda = True,
    test_deps = [
        "//sxt/algorithm/base:gather_mapper",
        "//sxt/algorithm/reduction:test_reducer",
        "//sxt/memory/resource:managed_device_resource",
        "//sxt/base/test:unit_test",
        "//sxt/execution/schedule:scheduler",
    ],
    deps = [
        ":block_computation_descriptor",
        ":completion",
        ":computation_setup",
        ":kernel",
        ":multiproduct_computation_descriptor",
        "//sxt/algorithm/base:reducer",
        "//sxt/base/container:span",
        "//sxt/base/device:memory_utility",
        "//sxt/base/device:stream",
        "//sxt/execution/async:future",
        "//sxt/execution/device:synchronization",
        "//sxt/memory/management:managed_array",
        "//sxt/memory/resource:async_device_resource",
        "//sxt/memory/resource:device_resource",
        "//sxt/memory/resource:pinned_resource",
    ],
)
