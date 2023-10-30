load(
    "//bazel:sxt_build_system.bzl",
    "sxt_cc_component",
)

sxt_cc_component(
    name = "combination_kernel",
    test_deps = [
        "//sxt/base/curve:example_element",
        "//sxt/base/test:unit_test",
        "//sxt/memory/management:managed_array",
        "//sxt/memory/resource:managed_device_resource",
        "//sxt/base/device:synchronization",
    ],
    deps = [
        "//sxt/base/curve:element",
        "//sxt/base/num:divide_up",
        "//sxt/base/num:log2p1",
    ],
)
